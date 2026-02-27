# GitHub 上传指南

> 本文档指导如何将整理好的实验记录上传到 GitHub

---

## 1. 创建 GitHub 仓库

### 1.1 在 GitHub 上创建新仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角 `+` → `New repository`
3. 填写信息：
   - **Repository name**: `bert-oelm` (建议)
   - **Description**: `BERT OELM: Head-wise Orthogonal Initialization for Efficient Transformer Fine-tuning`
   - **Visibility**: Public 或 Private (推荐 Public，便于分享)
   - **Initialize**: 不要勾选 (已有本地文件)
4. 点击 `Create repository`

---

## 2. 本地初始化并上传

### 2.1 进入项目目录

```bash
cd bert-oelm-paper
```

### 2.2 初始化 Git 仓库

```bash
git init
```

### 2.3 添加所有文件

```bash
git add .
```

### 2.4 提交文件

```bash
git commit -m "Initial commit: Complete BERT OELM experiments

- Add core implementation (modeling_bert_oelm.py, train_bert.py)
- Add experiment scripts and configs
- Add training logs for SST-2, MNLI, and ablation experiments
- Add timing analysis data
- Add complete documentation (README, EXPERIMENT_REPORT, EXPERIMENT_SUMMARY)"
```

### 2.5 连接远程仓库

```bash
# 替换 yourusername 为你的 GitHub 用户名
git remote add origin https://github.com/yourusername/bert-oelm.git
```

### 2.6 推送到 GitHub

```bash
git push -u origin main
# 或 git push -u origin master (取决于默认分支名)
```

---

## 3. 验证上传

### 3.1 检查文件大小

由于日志文件较大，确保不超过 GitHub 限制：

```bash
# 检查大文件
find . -type f -size +50M

# 如果存在超大文件，考虑使用 Git LFS
git lfs install
git lfs track "*.log"
git add .gitattributes
```

### 3.2 查看仓库

在浏览器中访问：
```
https://github.com/yourusername/bert-oelm
```

确认所有文件都已正确上传。

---

## 4. 设置仓库信息

### 4.1 添加 Topics (标签)

在 GitHub 仓库页面 → About → ⚙️ (齿轮图标)：
- `bert`
- `transformer`
- `orthogonal-initialization`
- `parameter-efficient-fine-tuning`
- `deep-learning`
- `nlp`
- `pytorch`

### 4.2 添加 Website

如果有相关论文页面或个人主页，可以添加：
```
https://yourwebsite.com/bert-oelm
```

### 4.3 启用 GitHub Pages (可选)

Settings → Pages → Source → Deploy from a branch → `main` / `docs`

---

## 5. 创建 Release (可选)

发布正式版本，便于引用：

1. 在 GitHub 仓库页面 → `Releases` → `Create a new release`
2. 填写信息：
   - **Tag version**: `v1.0.0`
   - **Release title**: `BERT OELM v1.0 - Complete Experiments`
   - **Description**:
```markdown
## Release v1.0.0

Complete implementation and experiments for BERT OELM paper.

### Features
- Head-wise orthogonal initialization for BERT
- Parameter-efficient fine-tuning (freeze Q/K)
- SST-2 and MNLI experiments
- Ablation study validating orthogonality necessity
- Fair comparison experiments with timing analysis

### Results
- SST-2: 91.28% (OELM) vs 93.12% (Baseline), gap -1.84%
- MNLI: 82.23% (OELM) vs 83.44% (Baseline), gap -1.21%
- Parameter reduction: 12.9%, Performance retention: 98.5%
```
3. 点击 `Publish release`

---

## 6. 完整命令速查

```bash
# 1. 进入目录
cd bert-oelm-paper

# 2. 初始化
git init

# 3. 添加文件
git add .

# 4. 提交
git commit -m "Initial commit: Complete BERT OELM experiments"

# 5. 连接远程
git remote add origin https://github.com/yourusername/bert-oelm.git

# 6. 推送
git push -u origin main

# 7. 后续更新
git add .
git commit -m "Update: description"
git push
```

---

## 7. 常见问题

### Q1: 推送失败 (Authentication failed)

**解决**: 使用 Personal Access Token 或 SSH

```bash
# 方法1: HTTPS + Token
git remote set-url origin https://TOKEN@github.com/yourusername/bert-oelm.git

# 方法2: SSH
git remote set-url origin git@github.com:yourusername/bert-oelm.git
```

### Q2: 文件太大无法推送

**解决**: 使用 Git LFS

```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "results/**/*.log"
git lfs track "results/**/*.json"

# 提交
git add .gitattributes
git add .
git commit -m "Add Git LFS for large files"
git push
```

### Q3: 日志文件太多

**解决**: 如果日志文件过大，可以：
1. 压缩后上传: `tar -czf results.tar.gz results/`
2. 或者只保留关键日志
3. 或者使用 Git LFS

---

## 8. 后续维护

### 8.1 定期更新

```bash
git add .
git commit -m "Update: new analysis/results"
git push
```

### 8.2 添加协作者

Settings → Manage access → Invite a collaborator

### 8.3 启用 Issues

用于讨论问题和追踪改进：
Settings → General → Features → ✅ Issues

---

## 9. 相关链接

- [GitHub Docs - Create a repo](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
- [GitHub Docs - Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

---

**完成！** 🎉

你的 BERT OELM 实验记录现在已经可以在 GitHub 上访问了！
