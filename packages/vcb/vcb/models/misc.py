from typing import TypeVar

# For now, simply a list of integers.
# If we run out of space, we may upgrade this to a bitmap.
IndexSet: TypeVar = list[int]
