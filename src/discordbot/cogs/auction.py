# 拍賣系統模組
# 此文件已重構，功能已分離到 _auction 資料夾中

from ._auction import setup

# 保持向後兼容性
__all__ = ['setup']