"""phase_2_intelligent_memory

Revision ID: c2b49b035a68
Revises: 84f31e60a4d2
Create Date: 2026-09-23 23:56:13.703169

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2b49b035a68'
down_revision: Union[str, Sequence[str], None] = '84f31e60a4d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add structured_claim to memories
    op.add_column('memories', sa.Column('structured_claim', sa.JSON(), nullable=True))
    # Add structured_claim to memory_versions
    op.add_column('memory_versions', sa.Column('structured_claim', sa.JSON(), nullable=True))

    # Create milestones table
    op.create_table(
        'milestones',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('project_id', sa.String(length=64), nullable=False),
        sa.Column('title', sa.String(length=256), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('milestone_type', sa.String(length=64), nullable=False),
        sa.Column('evidence_memory_id', sa.String(length=64), nullable=True),
        sa.Column('reached_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.ForeignKeyConstraint(['evidence_memory_id'], ['memories.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('milestones')
    op.drop_column('memory_versions', 'structured_claim')
    op.drop_column('memories', 'structured_claim')

