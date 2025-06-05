#!/bin/bash

# このスクリプトがあるディレクトリに移動
cd "$(dirname "$0")"

# 開発モードでインストール
pip install -e .
