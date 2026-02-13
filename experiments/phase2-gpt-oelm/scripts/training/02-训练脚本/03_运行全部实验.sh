#!/bin/bash
# =============================================================================
# 完整基准测试套件 (run_all.sh)
# =============================================================================
#
# 此脚本运行完整的基准测试套件，包括：
# 1. 标准Transformer基准 (Baseline)
# 2. 正交随机注意力 (实验组)
# 3. 随机高斯控制组
# 4. 生成综合对比报告
#
# 用法:
#   ./run_all.sh [output_dir]
#
# 参数:
#   output_dir - 结果输出目录 (默认: ./results)
#
# 示例:
#   ./run_all.sh
#   ./run_all.sh ./my_results
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 输出目录
OUTPUT_DIR="${1:-./results}"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  正交随机注意力 - 完整基准测试套件${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查脚本目录
SCRIPT_DIR="$(dirname "$0")"
if [ ! -f "$SCRIPT_DIR/run_baseline.sh" ]; then
    echo -e "${RED}错误: 找不到 run_baseline.sh${NC}"
    exit 1
fi

# =============================================================================
# 运行所有测试
# =============================================================================

# 1. 标准Transformer基准
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  [1/3] 运行标准Transformer基准${NC}"
echo -e "${BLUE}========================================${NC}"
bash "$SCRIPT_DIR/run_baseline.sh" "$OUTPUT_DIR/baseline"

# 2. 正交随机注意力
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  [2/3] 运行正交随机注意力${NC}"
echo -e "${CYAN}========================================${NC}"
bash "$SCRIPT_DIR/run_ortho_elm.sh" "$OUTPUT_DIR/ortho_elm"

# 3. 随机高斯控制组
echo -e "${MAGENTA}========================================${NC}"
echo -e "${MAGENTA}  [3/3] 运行随机高斯控制组${NC}"
echo -e "${MAGENTA}========================================${NC}"
bash "$SCRIPT_DIR/run_random_control.sh" "$OUTPUT_DIR/random_control"

# =============================================================================
# 生成综合报告
# =============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  生成综合对比报告${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

cat > "$OUTPUT_DIR/final_report.md" << EOF
# 正交随机注意力 - 综合基准测试报告

## 测试时间
$(date '+%Y-%m-%d %H:%M:%S')

## 测试概述

本报告汇总了三组基准测试的结果：

| 组别 | 注意力类型 | 说明 |
|------|-----------|------|
| Baseline | Standard Multi-Head Attention | 标准Transformer |
| 实验组 | Orthogonal Random Attention | 正交随机注意力 |
| 控制组 | Random Gaussian (Synthesizer) | 随机高斯控制 |

## 结果目录

- **Baseline**: \`$OUTPUT_DIR/baseline/\`
- **实验组**: \`$OUTPUT_DIR/ortho_elm/\`
- **控制组**: \`$OUTPUT_DIR/random_control/\`

## 快速开始

### 查看详细报告

\`\`\`bash
# Baseline报告
cat $OUTPUT_DIR/baseline/summary.md

# 实验组报告
cat $OUTPUT_DIR/ortho_elm/summary.md

# 控制组报告
cat $OUTPUT_DIR/random_control/summary.md

# 对比报告
cat $OUTPUT_DIR/ortho_elm/comparison/comparison_report.md
\`\`\`

### 查看JSON数据

\`\`\`bash
# 速度测试结果
python3 -m json.tool $OUTPUT_DIR/baseline/speed_results.json
python3 -m json.tool $OUTPUT_DIR/ortho_elm/speed_results.json
python3 -m json.tool $OUTPUT_DIR/random_control/speed_results.json

# 显存测试结果
python3 -m json.tool $OUTPUT_DIR/baseline/memory_results.json
python3 -m json.tool $OUTPUT_DIR/ortho_elm/memory_results.json
python3 -m json.tool $OUTPUT_DIR/random_control/memory_results.json
\`\`\`

## 测试方法

### 速度测试
- 训练速度: 测量单次前向+反向传播时间
- 推理速度: 测量单次前向传播时间
- 吞吐量: 计算每秒处理的tokens数量

### 显存测试
- 模型参数显存
- 前向传播峰值显存
- 反向传播峰值显存
- 完整训练步骤峰值显存

### 对比维度
1. **计算效率**: 速度对比
2. **内存效率**: 显存占用对比
3. **可扩展性**: 不同批次大小下的表现

## 核心发现

### 正交随机注意力的优势
1. **计算效率**: 通过随机特征近似降低复杂度
2. **内存效率**: 减少中间激活值存储
3. **数值稳定性**: 正交化提高稳定性

### 与标准Transformer的对比
- 速度提升: 待测试
- 显存节省: 待测试
- 精度保持: 待验证

### 与控制组的对比
- 正交化效果: 待验证
- 收敛稳定性: 待验证

## 后续分析

建议进行以下后续分析：

1. **精度对比**: 在下游任务上对比准确率
2. **收敛分析**: 对比训练收敛速度
3. **长序列测试**: 测试更长序列下的表现
4. **大规模测试**: 测试更大模型的表现

## 文件清单

\`\`\`
$OUTPUT_DIR/
├── baseline/
│   ├── speed_results.json
│   ├── memory_results.json
│   └── summary.md
├── ortho_elm/
│   ├── speed_results.json
│   ├── memory_results.json
│   ├── comparison/
│   │   ├── comparison_report.md
│   │   └── comparison_results.json
│   └── summary.md
├── random_control/
│   ├── speed_results.json
│   ├── memory_results.json
│   └── summary.md
└── final_report.md
\`\`\`

---

*本报告由基准测试套件自动生成*
EOF

echo -e "${GREEN}综合报告已保存到: $OUTPUT_DIR/final_report.md${NC}"
echo ""

# =============================================================================
# 完成
# =============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  完整基准测试套件运行完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "结果目录结构:"
tree -L 3 "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type f | head -20
echo ""
echo "查看综合报告:"
echo "  cat $OUTPUT_DIR/final_report.md"
echo ""
