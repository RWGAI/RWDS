
_base_ = '../dino/dino-5scale_swin-l_8xb2-36e_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Passenger Vehicle', 'Small Car', 'Bus', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck Tractor w/ Box Trailer', 'Trailer', 'Cargo/Container Car', 'Dump Truck', 'Front loader/Bulldozer', 'Shed', 'Building', 'Vehicle Lot', 'Shipping container lot', 'Shipping Container')
data_root='../../../RWDS_Dataset/RWDS_CZ/'


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/Group_combined_train_512_02.json',
        data_prefix=dict(img='train/Group_combined_train_images_512_02'),
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
        ann_file='val/Group_combined_val_512_02.json',
        data_prefix=dict(img='val/Group_combined_val_images_512_02'),
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
        ann_file='test/Group_combined_test_512_02.json',
        data_prefix=dict(img='test/Group_combined_test_images_512_02'),
        )
    )

load_from = 'https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
#load_from = "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"

model = dict(
    bbox_head=dict(
        num_classes=len(classes)))

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
    ann_file=data_root + 'val/Group_combined_val_512_02.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
metric='bbox',
classwise=True,
type='CocoMetric',
format_only=False,
backend_args=None,
 ann_file=data_root + 'test/Group_combined_test_512_02.json',)

#train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)