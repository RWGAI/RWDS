
_base_ = '../mask_rcnn/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py'
auto_scale_lr = dict(base_batch_size=16, enable=True)

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('no-damage', 'damaged')
data_root='../../../RWDS_Dataset/RWDS_HE/'


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/Hurricane_Group_harvey_train_512_02.json',
        data_prefix=dict(img='train/Hurricane_Group_harvey_train_images_512_02'),
        ),

    )

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/Hurricane_Group_harvey_val_512_02.json',
        data_prefix=dict(img='val/Hurricane_Group_harvey_val_images_512_02'),
        )
    )


test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/Hurricane_Group_harvey_test_512_02.json',
        data_prefix=dict(img='test/Hurricane_Group_harvey_test_images_512_02'),
        )
    )

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth'

#'https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
#load_from = "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"

model = dict(roi_head=dict(
    bbox_head=dict(
        num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes))))



num_classes=len(classes)

vis_backends = [
dict(type='LocalVisBackend'),
dict(type='TensorboardVisBackend')]

visualizer = dict(
name='visualizer',
type='DetLocalVisualizer',
vis_backends=vis_backends)

backend_args=None
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/Hurricane_Group_harvey_val_512_02.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
metric='bbox',
classwise=True,
type='CocoMetric',
format_only=False,
backend_args=None,
 ann_file=data_root + 'test/Hurricane_Group_harvey_test_512_02.json',)
