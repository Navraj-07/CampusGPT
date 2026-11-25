"""Authentication helpers and route protection decorator."""

import re
from functools import wraps
from typing import Callable

from flask import current_app, jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from database.db import get_user_by_id

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_REGEX = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def hash_password(password: str) -> str:
    """Hash a plaintext password using Werkzeug PBKDF2/scrypt."""
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Verify a plaintext password against its stored hash."""
    return check_password_hash(password_hash, password)


def validate_registration_input(
    full_name: str, username: str, email: str, password: str
) -> list[str]:
    """Validate user registration fields and return a list of error strings."""
    errors: list[str] = []
    if not full_name or len(full_name.strip()) < 2:
        errors.append("Full name must be at least 2 characters.")
    if not USERNAME_REGEX.match(username.strip()):
        errors.append(
            "Username must be 3–32 characters using letters, numbers, '.', '-', or '_'."
        )
    if not EMAIL_REGEX.match(email.strip()):
        errors.append("Please provide a valid email address.")
    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters long.")
    return errors


def login_required(view_func: Callable) -> Callable:
    """Require a valid logged-in session for HTML pages and JSON API endpoints."""

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        user_id = session.get("user_id")
        is_api = request.path.startswith("/api/") or request.is_json

        if not user_id:
            if is_api:
                return jsonify({"error": "Authentication required. Please log in."}), 401
            return redirect(url_for("login", next=request.path))

        user = get_user_by_id(current_app.config["DATABASE_PATH"], int(user_id))
        if not user:
            session.clear()
            if is_api:
                return jsonify({"error": "Session expired. Please log in again."}), 401
            return redirect(url_for("login"))

        return view_func(*args, **kwargs)

    return wrapped_view
