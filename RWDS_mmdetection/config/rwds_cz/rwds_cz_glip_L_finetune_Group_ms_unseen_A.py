
_base_ = '../glip//glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'

#auto_scale_lr = dict(base_batch_size=8, enable=True)

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Passenger Vehicle', 'Small Car', 'Bus', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck Tractor w/ Box Trailer', 'Trailer', 'Cargo/Container Car', 'Dump Truck', 'Front loader/Bulldozer', 'Shed', 'Building', 'Vehicle Lot', 'Shipping container lot', 'Shipping Container')
data_root='../../../RWDS_Dataset/RWDS_CZ/'


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/Group_ms_unseen_group_A_train_512_02.json',
        data_prefix=dict(img='train/Group_ms_unseen_group_A_train_images_512_02'),
        ),

    )

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/Group_ms_unseen_group_A_val_512_02.json',
        data_prefix=dict(img='val/Group_ms_unseen_group_A_val_images_512_02'),
        )
    )


test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/Group_ms_unseen_group_A_test_512_02.json',
        data_prefix=dict(img='test/Group_ms_unseen_group_A_test_images_512_02'),
        )
    )


load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800-e9be4274.pth'

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
    ann_file=data_root + 'val/Group_ms_unseen_group_A_val_512_02.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
metric='bbox',
classwise=True,
type='CocoMetric',
format_only=False,
backend_args=None,
 ann_file=data_root + 'test/Group_ms_unseen_group_A_test_512_02.json',)
