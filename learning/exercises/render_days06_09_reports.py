from pathlib import Path
import json,os
root=Path(__file__).resolve().parent
for day in ['day06','day07','day08','day09']:
    folder=root/day
    d=json.loads((folder/'results/results.json').read_text())
    s=f'# {day} 成果与实测结果\n\n自动测试通过；使用普通用户 zzy。GPU：{d["gpu"]}，CUDA {d["cuda"]}，PyTorch {d["torch"]}。\n\n'
    s+='## 交付与验收\n\n'
    descriptions={
      'day06':'修复编译拼写、64 位索引、参数边界、device guard 和 launch check；保留 shuffle + atomic 主体。原始文件已备份。',
      'day07':'交付独立 add 参考实现、可控归约 bug 与修复、首周复盘笔记。256 个全 1 的行归约：bug 版本得到 32，修复版得到 256。个人限时默写和口述仍需学习者自测。',
      'day08':'交付读合并/写合并 naive 转置、32×32 shared tile、flat copy 参考。计划的三个 shape 及更多边界输入精确对照通过。',
      'day09':'对照只改变 shared 行跨度的 32×32 / 32×33 tile。交付逐 lane bank CSV 和 SVG 图。正确性精确对照通过。'}
    s+=descriptions[day]+'\n\n'
    s+='```json\n'+json.dumps(d['checks'],ensure_ascii=False,indent=2)+'\n```\n\n'
    s+='## 本次性能实验\n\n'+d['timing']+'。首次 JIT/输入初始化不计时；这是一次热缓存实验，提交间隙、频率与运行负载可能影响小 kernel。\n\n'
    if day in ['day06','day07']:
        s+='| N | block | torch ms | 自写 ms | 相对 torch |\n|---:|---:|---:|---:|---:|\n'
        for r in d['rows']:
            s+=f'| {r["n"]} | {r["block"]} | {r["torch_ms"]:.6f} | {r["custom_ms"]:.6f} | {r["speedup"]:.3f}x |\n'
        if day=='day06':
            s+='\n自写归约包含分配、清零、atomic 累加；torch.sum 同样返回新输出。相同 max_blocks 不等于每个 block 相同工作量。当前简单实现未必比成熟库快，不能只挑最快配置作为普遍收益。随机数使用 FP64 对照与规模相关容差，atomic 累加不保证每次位级相同。\n'
        else:
            s+='\nadd 使用预分配输出，torch 使用 out=。大 N 时接近的时间不证明存在稳定优化收益；此日主要成果是从简单实现出发验证与定位归约 bug。\n'
    else:
        s+='| shape | 版本 | torch ms | 自写 ms | 相对 torch |\n|---|---|---:|---:|---:|\n'
        for r in d['rows']:
            s+=f'| {r["rows"]}×{r["cols"]} | {r["version"]} | {r["torch_ms"]:.6f} | {r["custom_ms"]:.6f} | {r["speedup_vs_torch"]:.3f}x |\n'
        s+='\nPyTorch 对照为将 a.t() 拷贝进预分配连续输出；a.t() 本身只产生视图。有效带宽包含缓存影响，不等于实测物理显存带宽。\n'
    if day=='day08':
        s+='\n### copy 参考\n\n| shape | flat copy ms |\n|---|---:|\n'
        for r in d['rows'][::3]:s+=f'| {r["rows"]}×{r["cols"]} | {r["copy_ms"]:.6f} |\n'
        s+='\nnaive_read 输入连续但输出跨行；naive_write 则相反。shared tile 让 global 两侧都访问行，但 tile 列读取仍有 bank conflict，下一日继续优化。copy 不是转置，仅作数据搬运参考。\n'
    if day=='day09':
        s+='\n### padding 对照\n\n| shape | 无 padding ms | padding ms | 相对无 padding |\n|---|---:|---:|---:|\n'
        for a,b in zip(d['rows'][::2],d['rows'][1::2]):
            s+=f'| {a["rows"]}×{a["cols"]} | {a["custom_ms"]:.6f} | {b["custom_ms"]:.6f} | {b["speedup_vs_unpadded"]:.3f}x |\n'
        s+='\n[bank 映射图](results/bank_mapping.svg) 和 [bank 映射 CSV](results/bank_mapping.csv) 展示理论地址映射。float 元素的 bank=(row*stride+col)%32；列读取时 stride32 对应同 bank 不同字，stride33 分散到全部 32 bank。shared 内存增加 128 字节。\n'
        log=folder/'results/ncu.log';code=folder/'results/ncu_exit.txt'
        if log.exists() and code.exists():
            text=log.read_text(errors='replace');exitcode=code.read_text().strip()
            if exitcode=='0':s+='\nNsight Compute 采集成功，详见 results/ncu.log；Event 表格独立于 profiler 运行。\n'
            elif 'ERR_NVGPUCTRPERM' in text:s+='\nNsight Compute 返回 ERR_NVGPUCTRPERM：Windows 未授权 GPU 性能计数器。本次未采集硬件冲突计数；性能时间和理论映射不能代替硬件计数证据。步骤见 [Windows 工具说明](../../WINDOWS_TOOLS.md)。\n'
            else:s+=f'\nNsight Compute 采集未通过，退出码 {exitcode}，见 results/ncu.log。未声称已验证硬件计数器变化。\n'
    s+='\n## 复现与学习复盘\n\n运行命令和接口约束见本目录 README.md。原始数据在 results/results.json、results/benchmark.csv，命令日志为 results/test.log。自动检查通过不代表用户已独立掌握；可用 README 的解释题继续口述复盘。\n'
    p=folder/'RESULTS.md';p.write_text(s);os.chown(p,1000,1000)
