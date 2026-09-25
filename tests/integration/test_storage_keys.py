from uuid import UUID, uuid4

from app.storage.keys import build_storage_key


def test_build_storage_key_uses_logical_document_version_and_file_ids() -> None:
    logical_document_id = UUID("11111111-1111-1111-1111-111111111111")
    document_version_id = UUID("22222222-2222-2222-2222-222222222222")
    file_id = UUID("33333333-3333-3333-3333-333333333333")

    storage_key = build_storage_key(
        logical_document_id=logical_document_id,
        document_version_id=document_version_id,
        file_id=file_id,
    )

    assert (
        storage_key == "documents/"
        "11111111-1111-1111-1111-111111111111/"
        "22222222-2222-2222-2222-222222222222/"
        "33333333-3333-3333-3333-333333333333"
    )


def test_build_storage_key_is_deterministic() -> None:
    logical_document_id = uuid4()
    document_version_id = uuid4()
    file_id = uuid4()

    first_key = build_storage_key(
        logical_document_id=logical_document_id,
        document_version_id=document_version_id,
        file_id=file_id,
    )

    second_key = build_storage_key(
        logical_document_id=logical_document_id,
        document_version_id=document_version_id,
        file_id=file_id,
    )

    assert first_key == second_key


def test_different_file_ids_produce_different_storage_keys() -> None:
    logical_document_id = uuid4()
    document_version_id = uuid4()
    first_file_id = uuid4()
    second_file_id = uuid4()

    first_key = build_storage_key(
        logical_document_id=logical_document_id,
        document_version_id=document_version_id,
        file_id=first_file_id,
    )

    second_key = build_storage_key(
        logical_document_id=logical_document_id,
        document_version_id=document_version_id,
        file_id=second_file_id,
    )

    assert first_key != second_key
