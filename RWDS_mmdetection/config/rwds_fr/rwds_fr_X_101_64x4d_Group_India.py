
_base_ = '../tood/tood_x101-64x4d_fpn_ms-2x_coco.py'
auto_scale_lr = dict(base_batch_size=16, enable=True)

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('no-damage', 'damaged')
data_root='../../../RWDS_Dataset/RWDS_FR/'


train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/Flooding_Group_India_train_512_02.json',
        data_prefix=dict(img='train/Flooding_Group_India_train_images_512_02'),
        ),

    )

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/Flooding_Group_India_val_512_02.json',
        data_prefix=dict(img='val/Flooding_Group_India_val_images_512_02'),
        )
    )


test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/Flooding_Group_India_test_512_02.json',
        data_prefix=dict(img='test/Flooding_Group_India_test_images_512_02'),
        )
    )


load_from = 'https://download.openmmlab.com/mmdetection/v2.0/tood/tood_x101_64x4d_fpn_mstrain_2x_coco/tood_x101_64x4d_fpn_mstrain_2x_coco_20211211_003519-a4f36113.pth'

model = dict(
    bbox_head=dict(
        num_classes=len(classes)))

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
    ann_file=data_root + 'val/Flooding_Group_India_val_512_02.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
metric='bbox',
classwise=True,
type='CocoMetric',
format_only=False,
backend_args=None,
 ann_file=data_root + 'test/Flooding_Group_India_test_512_02.json',)
