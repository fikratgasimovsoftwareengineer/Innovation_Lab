import yaml
yaml_file = """names:
  - target
  - target
nc: 2
train: /tensorfl_lab/yolo/yolo/train/images
val:  /tensorfl_lab/yolo/yolo/val/images

"""
with open('/tensorfl_lab/yolo/update_vggFuoco.yaml', 'w') as f:
    f.write(yaml_file)