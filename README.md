# sw_pipeline

`sw_pipeline` 是一个统一的事件驱动空间天气流程项目，用同一套 YAML 配置串联：

- GNSS 原始观测下载与处理
- GNSS 网格产品导入与标准化
- GOLD 场景发现与出图
- OMNI 参数获取与三联图
- GOLD 底图叠加 GNSS 阈值图
- 指定测站单星 TEC / ROTI 四宫格图

详细介绍文档见：[docs/PROJECT_INTRODUCTION.md](docs/PROJECT_INTRODUCTION.md)

## CLI

```powershell
swp run event --event storm_20241010_11
swp run event --event storm_20241010_11 --include-fetch
swp fetch gnss-raw --event storm_20241010_11
swp process gnss --event storm_20241010_11
swp plot overlay --event storm_20241010_11
swp plot panel --event storm_20241010_11
swp clean workspace
swp clean run --event storm_20241010_11
swp migrate-legacy --from D:\Desktop\lzt_code\lzt_prj
swp migrate-legacy --from D:\Desktop\lzt_code\lzt_thesis_code\GNSSdraw
swp migrate-legacy --from D:\Desktop\lzt_code\lzt_thesis_code\OMNIdarw\outputs\data
swp migrate-legacy --from D:\Desktop\lzt_code\lzt_thesis_code\GOLDdraw
```

- `swp run event` 默认只执行 `process + plot`
- 只有显式传入 `--include-fetch` 时才会运行下载阶段
- `storage/cache/` 被视为只读输入资产，清理命令不会触碰它

## 配置

- 全局配置：`config/base.yaml`
- 事件配置：`config/events/<event_id>.yaml`
- 详细说明：`docs/PROJECT_INTRODUCTION.md`

## 目录

```text
storage/
  cache/
  archive/
    pre_refactor/
  runs/<event_id>/
    manifests/
    processed/
      gnss/
      gold/
      omni/
    products/
      grids/
    figures/
      gnss/
      gold/
      omni/
      overlays/
      panels/
      station_series/
```
