import pytorch_lightning as pl
from typing import Literal, Sequence, Tuple, Union

Point2D = Tuple[int, int]
Point3D = Tuple[int, int, int]
Box2D = Tuple[Point2D, Point2D]
Box3D = Tuple[Point3D, Point3D]
Colour = Union[str, Tuple[float, float, float]]
Crop2D = Tuple[Tuple[int, int], Tuple[int, int]]
ModelName = Tuple[str, str, str]
Model = pl.LightningModule
PatientID = Union[int, str]
PhysPoint2D = Tuple[float, float]
PhysPoint3D = Tuple[float, float, float]
PatientIDs = Union[Literal['all'], PatientID, Sequence[PatientID]]
PatientView = Literal['axial', 'sagittal', 'coronal'],
PatientRegions = Union[Literal['all'], str, Sequence[str]]
TrainingPartition = Literal['train', 'validation', 'test']
ImageSize2D = Tuple[int, int]
ImageSize3D = Tuple[int, int, int]
ImageSpacing2D = Tuple[float, float]
ImageSpacing3D = Tuple[float, float, float]
TrainInterval = Union[int, str]
