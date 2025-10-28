# LEAF-YOLO Model Architecture

## Overview
LEAF-YOLO is a lightweight, edge-real-time object detection model specifically designed for small object detection in aerial imagery. It builds upon YOLOv7 and YOLOv5 architectures, introducing novel components for efficient multiscale feature extraction while maintaining high accuracy with low computational cost.

The model comes in two variants:
- **LEAF-YOLO-N (Nano)**: 1.2M parameters, optimized for ultra-low resource environments
- **LEAF-YOLO (Standard)**: 4.28M parameters, balancing performance and efficiency

## Core Innovation: Lightweight-Efficient Aggregating Fusion (LEAF)
LEAF-YOLO introduces the LEAF mechanism, which enhances multiscale feature extraction through:
- Efficient feature aggregation across different scales
- Reduced computational complexity
- Improved small object detection capabilities

## Architecture Components

### Backbone
The backbone follows a hierarchical structure similar to YOLOv7-tiny but incorporates LEAF-specific modifications:

#### Input Processing
- **Input**: RGB images (typically 640x640)
- **Initial Convolution**: Conv(32, 3x3, stride=2) → P1 (320x320)

#### Stage 1: P2 Feature Extraction (320x320 → 160x160)
- Conv(64, 3x3, stride=2)
- GhostConv(64, 3x3)
- **LEAF Block**: Multiple PConv layers with Concat operations
- C3_Res2Block(64) - CSP block with Bottle2neck

#### Stage 2: P3 Feature Extraction (160x160 → 80x80)
- MaxPool + Conv operations
- GhostConv(64, 3x3, stride=2)
- **LEAF Block**: PConv layers with Concat
- C3_Res2Block(128)

#### Stage 3: P4 Feature Extraction (80x80 → 40x40)
- MaxPool + Conv operations
- GhostConv(128, 3x3, stride=2)
- **LEAF Block**: PConv layers with Concat
- C3_Res2Block(256)

#### Stage 4: P5 Feature Extraction (40x40 → 20x20)
- MaxPool + Conv operations
- GhostConv(256, 3x3, stride=2)
- **LEAF Block**: PConv layers with Concat
- C3_Res2Block(256)

### Neck (Feature Pyramid Network)
The neck performs multiscale feature fusion using a PANet-style architecture:

#### Spatial Pyramid Pooling
- **SPPRFEM**: Spatial Pyramid Pooling with RFEM (Receptive Field Enhancement Module)
- Combines features from P5 with different kernel sizes

#### Up-sampling Path
1. **P5 → P4 Fusion**:
   - CoordConvATT(128) for coordinate-aware attention
   - Upsample (2x nearest)
   - Concat with P4 features
   - C3_Res2Block(128)

2. **P4 → P3 Fusion**:
   - CoordConvATT(64)
   - Upsample (2x nearest)
   - Concat with P3 features
   - C3_Res2Block(64)

3. **P3 → P2 Fusion**:
   - CoordConvATT(64)
   - Upsample (2x nearest)
   - Concat with P2 features
   - C3_Res2Block(64)

#### Down-sampling Path
1. **P2 → P3**:
   - Conv operations with LEAF-style Concat
   - GhostConv(128, 3x3, stride=2)
   - Concat with P3

2. **P3 → P4**:
   - Conv operations with LEAF-style Concat
   - GhostConv(256, 3x3, stride=2)
   - Concat with P4

3. **P4 → P5**:
   - Conv operations
   - GhostConv(128, 3x3, stride=2)
   - Concat with P5

### Detection Head
- **IDetect**: Improved Detection head with implicit layers
- Processes 4 detection scales: P2, P3, P4, P5
- Uses anchor-based detection with custom anchor configurations
- Outputs: [batch, anchors, classes+5] where 5 = x,y,w,h,confidence

## Key Modules

### GhostConv
- Efficient convolution using "ghost" features
- Reduces parameters while maintaining representational capacity
- Formula: Split → Cheap operations → Concat

### PConv (Partial Convolution)
- Applies convolution to partial channels
- Reduces computation in dense layers
- Maintains feature diversity

### C3_Res2Block
- Cross-Stage Partial (CSP) block
- Contains Bottle2neck modules
- Enhances gradient flow and feature reuse

### CoordConvATT
- Coordinate Convolution with Attention
- Adds spatial coordinate information to features
- Integrated attention mechanism (CoordAtt)

### SPPRFEM
- Spatial Pyramid Pooling with RFEM
- Multi-scale receptive field enhancement
- Trident Convolution for diverse receptive fields

### IDetect
- Improved Detection head
- Implicit knowledge distillation
- Enhanced feature decoupling

## Model Specifications

### LEAF-YOLO-N (Nano)
- **Parameters**: 1.2M
- **FLOPs**: 5.6G
- **Input Size**: 640x640
- **Backbone Channels**: [32, 64, 128, 256, 256]
- **Neck Channels**: [128, 64, 64, 128, 256]

### LEAF-YOLO (Standard)
- **Parameters**: 4.28M
- **FLOPs**: 20.9G
- **Input Size**: 640x640
- **Backbone Channels**: [32, 64, 128, 256, 512]
- **Neck Channels**: [256, 128, 64, 128, 256]

## Performance Metrics (VisDrone2019-DET-val)
- **AP@50:95**: LEAF-YOLO-N: 21.9%, LEAF-YOLO: 28.2%
- **AP@50**: LEAF-YOLO-N: 39.7%, LEAF-YOLO: 48.3%
- **AP@S** (Small objects): LEAF-YOLO-N: 14.0%, LEAF-YOLO: 20.0%
- **Inference Time** (RTX 3090): LEAF-YOLO-N: 16.2ms, LEAF-YOLO: 21.7ms

## Edge Deployment
- **Jetson AGX Xavier**:
  - LEAF-YOLO-N: 56 FPS @ FP16
  - LEAF-YOLO: 32 FPS @ FP16
- Supports TensorRT, ONNX export
- Efficient NMS integration

## Training Details
- **Dataset**: VisDrone2019-DET
- **Input Size**: 640x640
- **Batch Size**: 16-128 (depending on GPU)
- **Epochs**: 1000
- **Optimizer**: SGD/AdamW
- **Loss**: Combination of classification, regression, and auxiliary losses

## References
- Based on YOLOv7 and YOLOv5 architectures
- Incorporates techniques from GhostNet, RepVGG, and CoordConv
- Custom LEAF mechanism for aerial small object detection</content>
<parameter name="filePath">/workspaces/codespaces-blank/structure.md
