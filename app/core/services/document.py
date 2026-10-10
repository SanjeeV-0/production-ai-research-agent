from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document, DocumentStatus
from app.core.repositories.document import DocumentRepository
from app.ingestion.normalizer import calculate_content_hash
from app.ingestion.schemas import DocumentInput

# Unique-constraint names this service knows how to recover from. Keeping
# these as named constants (rather than inline string literals scattered
# through the retry/recovery logic below) means a constraint rename in a
# future migration only needs to change one place, and a conflict on any
# OTHER constraint is never mistaken for one of these two expected races.
_CONTENT_HASH_CONSTRAINT = "uq_documents_content_hash"
_VERSION_NUMBER_CONSTRAINT = "uq_documents_logical_version"

_MAX_VERSION_ALLOCATION_ATTEMPTS = 5


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a dedup-aware ingest call.

    `matched=True` means an existing Document row was returned instead of
    creating a new one -- `updated_metadata_fields` lists which fields on
    that existing row were changed by this call (empty if nothing
    changed). `foreign=True` means the match belongs to a DIFFERENT logical
    document than the one the caller explicitly targeted (only meaningful
    for `ingest_as_new_version`); callers must never silently move or merge
    logical documents in that case -- they just report the match.
    """

    matched: bool
    updated_metadata_fields: list[str] = field(default_factory=list)
    foreign: bool = False


def _constraint_name(exc: IntegrityError) -> str | None:
    """Best-effort extraction of the violated constraint's name from a
    psycopg3-backed `IntegrityError`, so recovery logic can tell "this is
    the expected race we know how to handle" apart from "this is some other
    integrity failure that must not be silently retried or swallowed"."""

    diag = getattr(exc.orig, "diag", None)

    return getattr(diag, "constraint_name", None)


class DocumentService:
    """Application-level operations for research documents."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = DocumentRepository(session)

    def _apply_submitted_metadata(
        self,
        document: Document,
        document_input: DocumentInput,
    ) -> list[str]:
        """Apply explicitly submitted, non-empty metadata fields from
        `document_input` onto an EXISTING, already-matched `document` row.

        Metadata in this schema belongs to the individual version row
        (there is no separate logical-document-level metadata table) -- so
        this updates only the one matched version, never every version of
        its logical document. A field is only applied when the submitted
        value is present and non-empty; an omitted or blank field never
        clears an existing value, since this API defines no explicit
        "clear this field" operation. Content/blob identity (content_hash)
        is never touched here -- metadata changes can never change what a
        duplicate check later matches against.
        """

        updated_fields: list[str] = []

        def maybe_apply(field_name: str, new_value: str | None) -> None:
            if new_value is None:
                return

            normalized = new_value.strip() if isinstance(new_value, str) else new_value

            if normalized in (None, ""):
                return

            if getattr(document, field_name) != normalized:
                setattr(document, field_name, normalized)
                updated_fields.append(field_name)

        maybe_apply("title", document_input.title)
        maybe_apply("document_type", document_input.document_type)
        maybe_apply("source", document_input.source)
        maybe_apply("authors", document_input.authors)

        if (
            document_input.publication_date is not None
            and document.publication_date != document_input.publication_date
        ):
            document.publication_date = document_input.publication_date
            updated_fields.append("publication_date")

        return updated_fields

    async def _return_duplicate(
        self,
        existing: Document,
        document_input: DocumentInput,
        foreign: bool = False,
    ) -> tuple[Document, MatchResult]:
        """Apply submitted metadata (if applicable) to an existing matched
        document and commit that change as its own small transaction,
        separate from -- and never blocking -- the ingestion decision
        itself."""

        updated_fields = (
            [] if foreign else self._apply_submitted_metadata(existing, document_input)
        )

        if updated_fields:
            existing = await self.repository.update(existing)

        await self.session.commit()

        return existing, MatchResult(
            matched=True,
            updated_metadata_fields=updated_fields,
            foreign=foreign,
        )

    async def ingest_as_new_or_duplicate(
        self,
        document_input: DocumentInput,
    ) -> tuple[Document, MatchResult]:
        """Ingest with NO caller-supplied logical document -- detect a
        global normalized-content duplicate (regardless of filename,
        category, or which logical document it belongs to) or create a
        brand new logical document with a server-generated ID.

        Global duplicate-content uniqueness is enforced at the database
        level by the `uq_documents_content_hash` constraint (on
        `content_hash` alone, not scoped to any logical document) -- the
        `get_by_content_hash` pre-check below is an optimization to avoid an
        unnecessary INSERT attempt in the common case, NOT the source of
        truth for uniqueness. A concurrent duplicate that slips past the
        pre-check is still caught by the constraint and recovered here by
        re-querying the authoritative row, never surfaced as an unhandled
        integrity error.
        """

        content_hash = calculate_content_hash(document_input.content)

        existing = await self.repository.get_by_content_hash(content_hash)

        if existing is not None:
            return await self._return_duplicate(existing, document_input)

        document = Document(
            title=document_input.title,
            authors=document_input.authors,
            source=document_input.source,
            publication_date=document_input.publication_date,
            document_type=document_input.document_type,
            logical_document_id=uuid4(),
            content_hash=content_hash,
            version_number=1,
            is_current=False,
            document_metadata={
                "content_length": len(document_input.content),
            },
        )

        try:
            document = await self.repository.create(document)
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()

            if _constraint_name(exc) != _CONTENT_HASH_CONSTRAINT:
                raise

            # Lost a race against a concurrent identical upload -- the
            # winner's row is now visible; return it as the authoritative
            # duplicate rather than surfacing this as a failed request.
            existing = await self.repository.get_by_content_hash(content_hash)

            if existing is None:
                raise

            return await self._return_duplicate(existing, document_input)

        return document, MatchResult(matched=False)

    async def ingest_as_new_version(
        self,
        logical_document_id: UUID,
        document_input: DocumentInput,
    ) -> tuple[Document, MatchResult]:
        """Add a revision to an explicitly chosen EXISTING logical document.

        If the submitted content already matches an existing version
        ANYWHERE (not just under `logical_document_id`), that global match
        is returned instead of creating an ambiguous new version -- per the
        approved policy, this never silently merges or moves either logical
        document; it simply reports the authoritative existing match
        (`MatchResult.foreign=True` when that match belongs to a different
        logical document than the one requested).

        Version-number allocation is retried (bounded) on an expected
        `uq_documents_logical_version` conflict -- see the module-level
        constant for the attempt cap and `_constraint_name` for how an
        unrelated integrity failure is told apart and re-raised instead of
        retried.
        """

        content_hash = calculate_content_hash(document_input.content)

        existing = await self.repository.get_by_content_hash(content_hash)

        if existing is not None:
            return await self._return_duplicate(
                existing,
                document_input,
                foreign=existing.logical_document_id != logical_document_id,
            )

        last_error: IntegrityError | None = None

        for _attempt in range(_MAX_VERSION_ALLOCATION_ATTEMPTS):
            latest_version_number = await self.repository.get_latest_version_number(
                logical_document_id
            )

            document = Document(
                title=document_input.title,
                authors=document_input.authors,
                source=document_input.source,
                publication_date=document_input.publication_date,
                document_type=document_input.document_type,
                logical_document_id=logical_document_id,
                content_hash=content_hash,
                version_number=latest_version_number + 1,
                is_current=False,
                document_metadata={
                    "content_length": len(document_input.content),
                },
            )

            try:
                document = await self.repository.create(document)
                await self.session.commit()

                return document, MatchResult(matched=False)

            except IntegrityError as exc:
                await self.session.rollback()

                constraint = _constraint_name(exc)

                if constraint == _CONTENT_HASH_CONSTRAINT:
                    # Another request created a matching-content row (under
                    # this or another logical document) between our
                    # pre-check and this insert -- same recovery as the
                    # upfront check above.
                    existing = await self.repository.get_by_content_hash(content_hash)

                    if existing is None:
                        raise

                    return await self._return_duplicate(
                        existing,
                        document_input,
                        foreign=existing.logical_document_id != logical_document_id,
                    )

                if constraint != _VERSION_NUMBER_CONSTRAINT:
                    raise

                # Expected version-number race: another concurrent upload
                # under the same logical document claimed this version
                # number first. Retry with a freshly re-read max() -- never
                # retried blindly with the same number.
                last_error = exc
                continue

        raise RuntimeError(
            "Could not allocate a version number for logical document "
            f"{logical_document_id} after {_MAX_VERSION_ALLOCATION_ATTEMPTS} attempts "
            "due to sustained concurrent contention."
        ) from last_error

    async def mark_processing(
        self,
        document: Document,
    ) -> Document:
        """Mark a document version as actively processing."""

        document.status = DocumentStatus.PROCESSING
        document.processing_attempt += 1
        document.processing_started_at = datetime.now(UTC)
        document.processing_completed_at = None
        document.failed_at = None
        document.last_error = None

        return await self.repository.update(document)

    async def mark_ready(
        self,
        document: Document,
    ) -> Document:
        """Mark a version READY.

        Only a logical document's VERY FIRST version (`version_number ==
        1`) is automatically promoted to current here -- this is the one
        case where reaching READY is allowed to change which version is
        current, since there is no prior current version a user could ever
        have intentionally kept. Every later version (`version_number > 1`)
        reaching READY leaves `is_current` untouched (still False from
        creation): a successfully ingested revision never displaces
        whatever is currently selected merely by finishing ingestion --
        the user must explicitly call `set_current` to promote it.
        """

        now = datetime.now(UTC)

        document.status = DocumentStatus.READY
        document.processing_completed_at = now
        document.failed_at = None
        document.last_error = None

        if document.version_number == 1:
            current_document = await self.repository.get_current_version(
                document.logical_document_id
            )

            if current_document is not None and current_document.id != document.id:
                current_document.is_current = False
                await self.repository.update(current_document)

            document.is_current = True

        return await self.repository.update(document)

    async def mark_failed(
        self,
        document: Document,
        error: str,
    ) -> Document:
        """Mark a document version as failed during processing.

        Never touches any OTHER version's `is_current` -- a failed version
        (first or later) can never displace whatever is currently current.
        """

        now = datetime.now(UTC)

        document.status = DocumentStatus.FAILED
        document.is_current = False
        document.processing_completed_at = now
        document.failed_at = now
        document.last_error = error

        return await self.repository.update(document)

    async def set_current(
        self,
        document: Document,
    ) -> Document:
        """Make `document` the current version of its logical document.

        Only a READY version may become current (raises `ValueError`
        otherwise -- the route maps this to 409). Already-current is a
        no-op success (idempotent). Demoting the previous current version
        (if any, and if different) and promoting this one happen as flushes
        within the same transaction, committed together, so a failure
        partway through leaves nothing committed and never two current
        versions. A concurrent promotion of a DIFFERENT version for the
        same logical document is still resolved correctly even though
        neither caller takes an explicit lock: the database's own
        `ix_documents_current_version` partial unique index is the actual
        enforcement point, and a conflict there is a genuine, expected race
        -- callers hitting it should re-fetch and retry, exactly like the
        content-hash/version-number races above.
        """

        if document.status != DocumentStatus.READY:
            raise ValueError(
                f"Only READY versions can be made current (status is {document.status})."
            )

        if document.is_current:
            return document

        current_document = await self.repository.get_current_version(document.logical_document_id)

        if current_document is not None and current_document.id != document.id:
            current_document.is_current = False
            await self.repository.update(current_document)

        document.is_current = True
        document = await self.repository.update(document)

        await self.session.commit()

        return document

    async def delete_version(
        self,
        document: Document,
    ) -> Document | None:
        """Delete a document version and promote the newest READY version."""

        was_current = document.is_current
        logical_document_id = document.logical_document_id
        document_id = document.id

        await self.repository.delete(document)

        if not was_current:
            return None

        replacement = await self.repository.get_newest_ready_version(
            logical_document_id=logical_document_id,
            exclude_document_id=document_id,
        )

        if replacement is None:
            return None

        replacement.is_current = True
        await self.repository.update(replacement)

        return replacement
