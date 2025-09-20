param()

$ErrorActionPreference = 'Stop'

# Create per-symbol directories
New-Item -ItemType Directory -Force 'qqq' | Out-Null
New-Item -ItemType Directory -Force 'spy' | Out-Null
New-Item -ItemType Directory -Force '^gspc' | Out-Null

# QQQ: move and rename strategy images
$qqqPngs = Get-ChildItem -File 'strategy_tqqq_reserve_*.png' -ErrorAction SilentlyContinue
foreach ($f in $qqqPngs) {
  $newName = ($f.Name -replace '^strategy_tqqq_', 'strategy_qqq_')
  Move-Item -LiteralPath $f.FullName -Destination (Join-Path -Path 'qqq' -ChildPath $newName) -Force
}

# QQQ: move other root artifacts if present
if (Test-Path 'strategy_tqqq_reserve_debug.csv') {
  Move-Item -LiteralPath 'strategy_tqqq_reserve_debug.csv' -Destination 'qqq/strategy_qqq_reserve_debug.csv' -Force
}
if (Test-Path 'nasdaq_temperature.png') {
  Move-Item -LiteralPath 'nasdaq_temperature.png' -Destination 'qqq/' -Force
}
if (Test-Path 'fit_constant_growth.png') {
  Move-Item -LiteralPath 'fit_constant_growth.png' -Destination 'qqq/' -Force
}

# SPY: move artifacts
$spyPngs = Get-ChildItem -File 'strategy_spy_reserve*.png' -ErrorAction SilentlyContinue
foreach ($f in $spyPngs) {
  Move-Item -LiteralPath $f.FullName -Destination 'spy/' -Force
}
if (Test-Path 'strategy_spy_reserve_debug.csv') {
  Move-Item -LiteralPath 'strategy_spy_reserve_debug.csv' -Destination 'spy/' -Force
}
if (Test-Path 'fit_constant_growth_SPY.png') {
  Move-Item -LiteralPath 'fit_constant_growth_SPY.png' -Destination 'spy/' -Force
}
if (Test-Path 'temperature_SPY.png') {
  Move-Item -LiteralPath 'temperature_SPY.png' -Destination 'spy/' -Force
}

# ^GSPC: move artifacts
foreach ($name in @('fit_constant_growth_^GSPC.png','strategy_gspc_reserve_A1_^GSPC.png','temperature_^GSPC.png','strategy_gspc_reserve_A1.csv')) {
  if (Test-Path $name) {
    Move-Item -LiteralPath $name -Destination '^gspc' -Force
  }
}

Write-Output 'Artifact organization complete.'


