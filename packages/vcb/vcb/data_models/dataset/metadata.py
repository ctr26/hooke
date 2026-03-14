from pydantic import BaseModel


class DatasetMetadata(BaseModel):
    """
    A dataset metadata object.

    Note (cwognum): There is additional metadata that is not included here, since it's not used in this code base.
    """

    dataset_id: str
    biological_context: set[str]
