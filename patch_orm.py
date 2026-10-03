import sys

with open('smriti/models.py', 'r', encoding='utf-8') as f:
    c = f.read()

type_decorator_code = '''from sqlalchemy.types import TypeDecorator, JSON

class VectorType(TypeDecorator):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql' and HAS_PGVECTOR:
            return dialect.type_descriptor(Vector())
        else:
            return dialect.type_descriptor(JSON())
'''

# Replace old implementation
import re
c = re.sub(r'    if HAS_PGVECTOR:\n        embedding = Column\(Vector\(\), nullable=True\)', r'    embedding = Column(VectorType, nullable=True)', c)

# Insert type decorator after HAS_PGVECTOR
c = c.replace('    HAS_PGVECTOR = False\n\nfrom sqlalchemy.orm', '    HAS_PGVECTOR = False\n\n' + type_decorator_code + '\nfrom sqlalchemy.orm')

with open('smriti/models.py', 'w', encoding='utf-8') as f:
    f.write(c)
