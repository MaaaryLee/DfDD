[CmdletBinding()]
param(
    [string]$BlenderPath = 'blender',
    [string]$Distribution = 'Ubuntu'
)

$ErrorActionPreference = 'Stop'
$sceneWorkspace = Split-Path -Parent $PSScriptRoot
$blenderExe = (Get-Command $BlenderPath -ErrorAction Stop).Source
& wsl -d $Distribution --cd $sceneWorkspace -- bash scripts/generate_hello_room.sh
if ($LASTEXITCODE -ne 0) {
    throw "Infinigen exited with code $LASTEXITCODE. See outputs/hello_room/logs."
}
& wsl -d $Distribution --cd (Join-Path $sceneWorkspace 'work/infinigen') -- .venv/bin/python ../../scripts/package_hello_room.py
if ($LASTEXITCODE -ne 0) {
    throw "Scene packaging failed with code $LASTEXITCODE."
}
$sceneFile = Join-Path $sceneWorkspace 'outputs/hello_room/hello_room.blend'
$frameScript = Join-Path $PSScriptRoot 'frame_hello_room.py'
& $blenderExe --background $sceneFile --python $frameScript
if ($LASTEXITCODE -ne 0) {
    throw "Preview rendering failed with code $LASTEXITCODE."
}
