# 代码版本管理

远端仓库：https://github.com/aihongjian777-crypto/BSP-cvpr

本仓库管理源码、运行脚本和说明文档。数据集、模型权重、实验输出、
临时文件、编辑器配置和本地密钥通过 `.gitignore` 排除，不会随代码上传。
`data/isic` 中的预处理 Python 脚本和上游许可文件保留在版本管理中。
该目录中的 `class_id.csv`、`ISIC_setting_1/isic_in_5_folds.json` 也是本地数据，
新机器运行 ISIC 实验时需另行准备；数据和权重路径见 `ISIC2018_SERVER.md`。

## 保存版本并同步 GitHub

在项目目录执行：

```powershell
git status
git diff
git add .
git diff --cached --stat
git commit -m "说明本次修改"
git push
```

每次提交都是一个可恢复的代码版本。提交后，推送才会把该版本备份到 GitHub。
仅编辑或保存文件不会自动创建 Git 提交。

## 查看与撤销

```powershell
# 查看最近版本和当前改动
git log --oneline -10
git diff

# 撤销指定提交：生成新的撤销提交，保留原有历史
# 将 COMMIT_ID 替换为实际提交编号；若有冲突，需要先解决冲突。
git revert COMMIT_ID
git push
```

撤销前先保存当前未提交的修改。也可以在聊天中说明需要撤销哪次修改，
由助手检查状态后操作。

Git 只能恢复已提交的内容；被忽略的数据、权重和实验结果应单独备份。
