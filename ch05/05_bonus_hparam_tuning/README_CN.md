# 预训练的超参数优化

[hparam_search.py](hparam_search.py) 脚本基于 [附录 D：为训练循环添加花里胡哨功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中的扩展训练函数，旨在通过网格搜索找到最佳超参数。

>[!NOTE]
>此脚本将需要很长时间运行。您可能需要减少在 `HPARAM_GRID` 字典顶部探索的超参数配置数量。