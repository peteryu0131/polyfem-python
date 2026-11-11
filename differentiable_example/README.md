# Differentiable PolyFEM examples

**开会看[案例与 loss 来源对照表](meeting/optimization_examples.md)。第一次用 API，从[材料梁](samples/material_and_contact/material_beam.py)开始。** 它用 198 个设计变量控制 198 个单元的刚度，打印一次更新前后的柔度。

从仓库根目录运行：

```powershell
conda run -n polyfem python -m differentiable_example.samples.material_and_contact.material_beam
```

| 你要做什么 | 打开这里 |
| --- | --- |
| 了解材料、初始条件、边界控制和形状接口，自己写 loss | [API 使用指南](guides/README.md) |
| 按问题找一个可以运行的例子 | [49 个 samples](samples/README.md) |
| 找老师某个优化目录的 Python 实现 | [18 个 optimizations](optimizations/README.md) |
| 看晶格撞击实验 | [正向仿真](paper_example/impact_forward.py)、[材料与梯度](paper_example/material_opt_explained.md)、[形状参数](paper_example/shape_opt_notes.md) |
| 复查梯度、原版对照或已有失败记录 | [维护者验证入口](../tools/validation/README.md) |
