#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从蛋白质序列表中提取 ProtMamba 模型的嵌入向量

读取包含两列的 CSV 文件：第一列为蛋白质名称，第二列为氨基酸序列。
脚本会加载预训练的 ProtMamba 模型，计算每个序列的嵌入，并将
结果保存为 ``npz`` 文件。
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import torch

# 显式从各子模块导入，避免因相对路径导致无法找到对象
from ProtMamba_ssm.utils import tokenizer
from ProtMamba_ssm.modules import load_model, MambaLMHeadModelwithPosids


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用预训练的 ProtMamba 模型提取蛋白质序列的特征向量"
    )
    parser.add_argument(
        "csv",
        help="包含蛋白质名称和序列的 CSV 文件路径"
    )
    parser.add_argument(
        "--model",
        default="data/example_pretrained_model",
        help="预训练模型所在目录"
    )
    parser.add_argument(
        "--output",
        default="embeddings.npz",
        help="保存嵌入向量的文件路径"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算所使用的设备"
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="如果 CSV 第一行是表头，则跳过该行",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    # 加载预训练模型
    model = load_model(
        args.model,
        device=device,
        model_class=MambaLMHeadModelwithPosids,
        dtype=torch.bfloat16,
        checkpoint_mixer=False,  # 推理时必须为 False
    )
    model.eval()

    names = []
    vectors = []

    csv_path = Path(args.csv)
    with csv_path.open() as f:
        reader = csv.reader(f)
        if args.skip_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            name, seq = row[0], row[1]
            names.append(name)

            # 将序列转换为模型输入的 token
            tokens = tokenizer([seq], concatenate=True)
            pos_ids = torch.arange(tokens.shape[1])[None, :]
            tokens = tokens.to(device)
            pos_ids = pos_ids.to(device)

            with torch.no_grad():
                # save_layer 指定要保存哪一层的隐藏状态，这里取最后一层
                hs = model(
                    input_ids=tokens,
                    position_ids=pos_ids,
                    save_layer=[model.config.n_layer],
                )
            # 取 <cls> token 对应的表示作为该序列的特征向量
            emb = hs[model.config.n_layer][0, 0].cpu().numpy()
            vectors.append(emb)

    vectors = np.stack(vectors)
    np.savez(args.output, names=np.array(names), embeddings=vectors)


if __name__ == "__main__":
    main()
