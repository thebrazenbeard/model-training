# Emergency restoration of the original PreActive tasks after R13 maintenance.
# Exact R13 train PID ownership only; never stop unrelated processes.
param([switch]$Emergency)
$ErrorActionPreference = 'Stop'
$root = 'D:\VERA\portal-task-state'
$log = Join-Path $root 'VERA_R13_PREACTIVE_RECOVERY_20261016.log'
$repo = 'D:\VERA\worktrees\vera-training-r13-behavior-correction-20261016'
$spec = Join-Path $repo 'successor\experiments\V10R3R13_FROZEN_BEHAVIOR_SPEC_20261016_V1.json'
$names = @('PreActive Window Supervisor','PreActive Qwen Endpoint','PreActive Daemon')
function Log([string]$value) {
  Add-Content -LiteralPath $log -Value ((Get-Date).ToString('o')+' '+$value)
}
try {
  if((Get-Date).ToString('yyyyMMdd') -ne '20261016') {
    Log 'DATE_NOT_AUTHORIZED_RESTORE_ACTION_SKIPPED'; exit 0
  }
  if($Emergency) {
    $trainers = @(Get-CimInstance Win32_Process | Where-Object {
      $_.Name -eq 'python.exe' -and $_.CommandLine -and
      $_.CommandLine.Contains($repo) -and $_.CommandLine.Contains($spec) -and
      $_.CommandLine -match 'successor\.experiments\.train_v10r2_dev'
    })
    foreach($trainer in $trainers) {
      Stop-Process -Id $trainer.ProcessId -Force -ErrorAction Stop
      Log ('EXACT_R13_SAFETY_STOP_PID='+$trainer.ProcessId)
    }
    Start-Sleep -Seconds 5
  }
  $allowed=@('Monday','Tuesday','Wednesday','Thursday','Friday')
  $now=Get-Date
  if(-not($allowed -contains $now.DayOfWeek.ToString()) -or
     $now.TimeOfDay -ge [TimeSpan]'13:59:00') {
    Log 'OUTSIDE_PREACTIVE_AUTHORIZED_WINDOW_NO_START';exit 0
  }
  foreach($name in $names) {
    $task=Get-ScheduledTask -TaskName $name -ErrorAction Stop
    if($task.State -ne 'Running') {Start-ScheduledTask -TaskName $name -ErrorAction Stop}
    Log ('RESTORE_REQUESTED '+$name)
  }
  for($i=0;$i -lt 65;$i++) {
    Start-Sleep -Seconds 3
    try {
      $state=@($names|ForEach-Object{(Get-ScheduledTask -TaskName $_ -ErrorAction Stop).State})
      if(@($state|Where-Object{$_ -ne 'Running'}).Count){continue}
      $answer=Invoke-RestMethod 'http://127.0.0.1:18081/v1/models' -TimeoutSec 3
      if(@($answer.data|ForEach-Object{$_.id}) -contains 'qwen3.5-4b-local') {
        Log 'PREACTIVE_RESTORED_AND_API_MODEL_ID_VERIFIED';exit 0
      }
    } catch {}
  }
  Log 'PREACTIVE_RESTORE_UNVERIFIED';exit 2
} catch {
  Log ('PREACTIVE_RESTORE_ERROR '+$_.Exception.Message);exit 2
}
