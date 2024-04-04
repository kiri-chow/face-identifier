# Face Identifier

## Info

This module includes two CNN models: `FaceDetector` and `FaceIdentifier`.

The detector aims to detect face in a given image and return the bounding box, the output format is `[has_face, x1, y1, x2, y2]`. A `CropFace` transformation wrapping the detector is also implemented.

The identifier returns a vector(face re-ID) after reading a face image.

## Usage

### Face Detection

```
from face_identifier.model import FaceDetector

model = FaceDetector.load("<path of state dict>")
predictions = model(images_tensor)  # every row is [has_face, x1, y1, x2, y2]
```python

### Face Cropping

```
from face_identifier.transforms import CropFace

cropper = CropFace("<path of state dict>")
image_cropped = cropper(image_tensor)  # returning a tensor
```python

### Face re-ID

```
from face_identifier.model import FaceIdentifier

model = FaceIdentifier.load("<path of state dict>")
face_id = model(images_tensor)
```

## Training

You can refer the notebooks to train your own model.

