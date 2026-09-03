"""Job roles and their question sets."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from ..database import get_db
from ..deps import get_current_user
from ..models import Interview, Question, Role, User
from ..schemas import RoleCreate, RoleOut, RoleSummary, RoleUpdate

router = APIRouter(prefix="/roles", tags=["roles"])


def _get_role(db: Session, role_id: int) -> Role:
    role = (
        db.query(Role)
        .options(joinedload(Role.questions))
        .filter(Role.id == role_id)
        .one_or_none()
    )
    if role is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    return role


@router.get("", response_model=list[RoleSummary])
def list_roles(
    include_inactive: bool = True,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[RoleSummary]:
    query = db.query(Role)
    if not include_inactive:
        query = query.filter(Role.is_active.is_(True))

    question_counts = dict(
        db.query(Question.role_id, func.count(Question.id)).group_by(Question.role_id).all()
    )
    interview_counts = dict(
        db.query(Interview.role_id, func.count(Interview.id)).group_by(Interview.role_id).all()
    )

    return [
        RoleSummary(
            id=role.id,
            title=role.title,
            department=role.department,
            is_active=role.is_active,
            question_count=question_counts.get(role.id, 0),
            interview_count=interview_counts.get(role.id, 0),
        )
        for role in query.order_by(Role.created_at.desc()).all()
    ]


@router.post("", response_model=RoleOut, status_code=status.HTTP_201_CREATED)
def create_role(
    payload: RoleCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Role:
    if not payload.questions:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Add at least one question")

    role = Role(
        title=payload.title.strip(),
        department=payload.department.strip(),
        description=payload.description.strip(),
        owner_id=user.id,
    )
    for index, question in enumerate(payload.questions):
        role.questions.append(
            Question(
                text=question.text.strip(),
                category=question.category or "general",
                time_limit_seconds=question.time_limit_seconds,
                order_index=index,
            )
        )
    db.add(role)
    db.commit()
    db.refresh(role)
    return role


@router.get("/{role_id}", response_model=RoleOut)
def get_role(role_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> Role:
    return _get_role(db, role_id)


@router.patch("/{role_id}", response_model=RoleOut)
def update_role(
    role_id: int,
    payload: RoleUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> Role:
    role = _get_role(db, role_id)

    if payload.title is not None:
        role.title = payload.title.strip()
    if payload.department is not None:
        role.department = payload.department.strip()
    if payload.description is not None:
        role.description = payload.description.strip()
    if payload.is_active is not None:
        role.is_active = payload.is_active

    if payload.questions is not None:
        if not payload.questions:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A role needs at least one question")
        used = db.query(Interview).filter(Interview.role_id == role.id).count()
        if used:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Questions cannot be changed once candidates have been invited. Create a new role instead.",
            )
        role.questions.clear()
        db.flush()
        for index, question in enumerate(payload.questions):
            role.questions.append(
                Question(
                    text=question.text.strip(),
                    category=question.category or "general",
                    time_limit_seconds=question.time_limit_seconds,
                    order_index=index,
                )
            )

    db.commit()
    db.refresh(role)
    return role


@router.delete("/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_role(role_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> None:
    role = _get_role(db, role_id)
    interviews = db.query(Interview).filter(Interview.role_id == role.id).count()
    if interviews:
        # Keep the audit trail: deactivate instead of destroying interview history.
        role.is_active = False
        db.commit()
        return
    db.delete(role)
    db.commit()
