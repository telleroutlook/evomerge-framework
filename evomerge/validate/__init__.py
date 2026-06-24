"""evomerge.validate — contamination and schema validation helpers."""
from evomerge.validate.contamination import check_contamination
from evomerge.validate.schema_check import validate_training_record

__all__ = ["check_contamination", "validate_training_record"]
