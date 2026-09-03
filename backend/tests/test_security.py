"""Tokens, password hashing, rate limiting and upload validation."""
from __future__ import annotations

import time

import pytest

from app import ratelimit
from app.security import (
    TokenError,
    create_access_token,
    create_invite_token,
    decode_token,
    hash_password,
    read_invite_token,
    verify_password,
)


def test_password_round_trip():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)
    assert not verify_password("anything", "")


def test_password_longer_than_bcrypt_limit():
    long_password = "a" * 200
    hashed = hash_password(long_password)
    assert verify_password(long_password, hashed)


def test_access_token_round_trip():
    token = create_access_token(7, "user@example.com")
    payload = decode_token(token, expected_type="access")
    assert payload["sub"] == "7"
    assert payload["email"] == "user@example.com"


def test_token_type_is_enforced():
    invite = create_invite_token("abc123")
    with pytest.raises(TokenError):
        decode_token(invite, expected_type="access")
    assert read_invite_token(invite) == "abc123"


def test_expired_token_is_rejected():
    token = create_invite_token("abc123", hours=0)
    time.sleep(1.1)
    with pytest.raises(TokenError):
        read_invite_token(token)


def test_tampered_token_is_rejected():
    token = create_invite_token("abc123")
    with pytest.raises(TokenError):
        read_invite_token(token[:-3] + "abc")


def test_rate_limiter_window():
    ratelimit.reset()
    key = "test-client"
    assert all(ratelimit.check(key, limit=3) for _ in range(3))
    assert ratelimit.check(key, limit=3) is False
    ratelimit.reset()
    assert ratelimit.check(key, limit=3) is True


def test_storage_rejects_bad_extension(tmp_path):
    import io

    from fastapi import UploadFile

    from app.services import storage

    upload = UploadFile(filename="virus.sh", file=io.BytesIO(b"#!/bin/sh"))
    with pytest.raises(storage.StorageError):
        storage.save_upload(upload, "test", [".wav"], max_mb=1)


def test_storage_enforces_size_limit():
    import io

    from fastapi import UploadFile

    from app.services import storage

    upload = UploadFile(filename="big.wav", file=io.BytesIO(b"0" * (2 * 1024 * 1024)))
    with pytest.raises(storage.StorageError) as exc:
        storage.save_upload(upload, "test", [".wav"], max_mb=1)
    assert "larger than" in str(exc.value)


def test_storage_sanitises_filenames():
    from app.services.storage import safe_name

    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("my resume (final).pdf") == "my_resume__final_.pdf"
    assert safe_name("") == "upload"


def test_storage_keeps_nested_interview_folders(fake_audio):
    """Recordings live under uploads/interviews/<public_id>/ so cleanup can find them."""
    import io
    import os

    from fastapi import UploadFile

    from app.services import storage

    public_id = "abc123def456"
    upload = UploadFile(filename="answer.wav", file=io.BytesIO(fake_audio))
    path = storage.save_audio(upload, public_id, question_id=7)

    assert os.path.basename(path) == "q7.wav"
    assert os.path.basename(os.path.dirname(path)) == public_id
    assert os.path.basename(os.path.dirname(os.path.dirname(path))) == "interviews"
    assert os.path.exists(path)

    storage.delete_interview_files(public_id)
    assert not os.path.exists(path)


def test_storage_blocks_path_traversal_in_subdir(fake_audio):
    import io
    import os

    from fastapi import UploadFile

    from app.services import storage

    upload = UploadFile(filename="answer.wav", file=io.BytesIO(fake_audio))
    path = storage.save_audio(upload, "../../etc/evil", question_id=1)
    assert os.path.abspath(path).startswith(os.path.abspath(storage._root()))
