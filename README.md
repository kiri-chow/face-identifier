# Face Identifier

## Info

This module includes two CNN models: `FaceDetector` and `FaceIdentifier`.

The detector aims to detect face in a given image and return the bounding box, the output format is `[has_face, x1, y1, x2, y2]`. A `CropFace` transformation wrapping the detector is also implemented.

The identifier returns a vector(face re-ID) after reading a face image.

## Dataset

The training and evaluation are based on [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) .

## Usage

### Face re-ID

```python
from face_identifier.model import FaceIdentifier

model = FaceIdentifier.load("<path of state dict>")
face_id = model(images_tensor)
```

### Face Detection

```python
from face_identifier.model import FaceDetector

model = FaceDetector.load("<path of state dict>")
predictions = model(images_tensor)  # every row is [has_face, x1, y1, x2, y2]
```

### Face Cropping

```python
from face_identifier.transforms import CropFace

cropper = CropFace("<path of state dict>")
image_cropped = cropper(image_tensor)  # returning a tensor
```

## Training

You can refer the notebooks to train your own model.

