#!/bin/bash

python main.py --base configs/xxxxx.yaml --train --trainer.devices 8 --logdir /xxxx/xxx/ --extra 'model.base_learning_rate=5.e-6,model.params.backbone_config.params.dropout=0.08'