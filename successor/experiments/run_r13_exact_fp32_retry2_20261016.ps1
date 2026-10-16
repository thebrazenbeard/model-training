# Reversible PreActive maintenance and bounded R13 training controller.
# Separate one-shot SYSTEM watchdog restores PreActive if this process is lost.
param(
  [Parameter(Mandatory=$true)][string]$ExpectedHead,
  [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
$repo='D:\VERA\worktrees\vera-training-r13-behavior-correction-20261016'
$state='D:\VERA\portal-task-state'
$spec=Join-Path $repo 'successor\experiments\V10R3R13_FROZEN_BEHAVIOR_SPEC_20261016_RETRY2.json'
$corpus='D:\VERA\.scratch\vera-v10r3r13-voice-20261016-v1\train.jsonl'
$parent='D:\VERA\models\adapters\v10r3r12-vera-fp32-qwen35-r11-resume-20261015-trl-retry2\adapter\adapter_model.safetensors'
$output='D:\VERA\models\adapters\v10r3r13-vera-identity-judgment-qwen35-r12-resume-20261016-retry2'
$python='C:\ProgramData\ProRun\model-env\Scripts\python.exe'
$recover=Join-Path $repo 'successor\experiments\restore_preactive_r13_retry2_20261016.ps1'
$taskName='BT2 Vera R13 Retry2 Recovery 20261016'
$gate=Join-Path $state 'VERA_R13_BEHAVIOR_LAUNCH_GATE_20261016_V4.json'
$stdout=Join-Path $state 'VERA_R13_TRAIN_20261016_V4_STDOUT.log'
$stderr=Join-Path $state 'VERA_R13_TRAIN_20261016_V4_STDERR.log'
$log=Join-Path $state 'VERA_R13_TRAIN_20261016_CONTROLLER_V4.log'
$receipt=Join-Path $state 'VERA_R13_TRAIN_20261016_CONTROLLER_V4.json'
$specSha='0127d7e88f3ed608117d2a1388d0c82a3d921df75fbf408374db3945b1060418'
$corpusSha='90de3879113a99f85002ec884d504ee487e5ed699d9deaa0146290c47597b3c5'
$parentSha='a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c'
$names=@('PreActive Window Supervisor','PreActive Qwen Endpoint','PreActive Daemon')
$paused=$false
$restored=$false
$watchRegistered=$false
$trainerPid=$null
$trainerExited=$false
$trainerExit=$null
$stateResult='NOT_STARTED'
$reason=''
$start=[DateTimeOffset]::Now
function Log([string]$msg){
 $line=[DateTimeOffset]::Now.ToString('o')+' '+$msg
 Add-Content -LiteralPath $log -Value $line
 Write-Output $line
}
function Hash([string]$file) {
 if(-not(Test-Path $file -PathType Leaf)){throw ('MISSING '+$file)}
 return (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
}
function OwnedR13 {
 return @(Get-CimInstance Win32_Process | Where-Object {
   $_.Name -eq 'python.exe' -and $_.CommandLine -and
   $_.CommandLine.Contains($repo) -and
   $_.CommandLine.Contains($spec) -and
   $_.CommandLine -match 'successor\.experiments\.train_v10r2_dev'
 })
}
function StopOwnedR13([string]$why){
 Log ('STOP_EXACT_R13_REASON '+$why)
 foreach($item in @(OwnedR13)) {
  Stop-Process -Id $item.ProcessId -Force -ErrorAction Stop
  Log ('STOPPED_EXACT_R13_TRAINER '+$item.ProcessId)
 }
 if($null -ne $trainerPid){
  try{Stop-Process -Id $trainerPid -Force -ErrorAction Stop}catch{}
 }
}
function RestorePreActive {
 if(-not $paused){return}
 Log 'PREACTIVE_RESTORE_BEGIN'
 # Restoration only when inside the user-owned autonomy window.
 $allowed=@('Monday','Tuesday','Wednesday','Thursday','Friday')
 $now=Get-Date
 if(-not($allowed -contains $now.DayOfWeek.ToString()) -or
   $now.TimeOfDay -ge [TimeSpan]'13:59:00'){
   Log 'OUTSIDE_WINDOW_RESTORE_NOT_AUTHORIZED'
   return
 }
 foreach($name in $names){
  try{
   $task=Get-ScheduledTask -TaskName $name -ErrorAction Stop
   if($task.State -ne 'Running'){Start-ScheduledTask -TaskName $name -ErrorAction Stop}
   Log ('RESTORE_TASK_REQUESTED '+$name)
  }catch{Log ('RESTORE_TASK_ERROR '+$name+' '+$_.Exception.Message)}
 }
 for($i=0;$i -lt 65;$i++){
  Start-Sleep -Seconds 3
  try{
   $states=@($names|ForEach-Object{(Get-ScheduledTask -TaskName $_ -ErrorAction Stop).State})
   if(@($states|Where-Object{$_ -ne 'Running'}).Count){continue}
   $resp=Invoke-RestMethod 'http://127.0.0.1:18081/v1/models' -TimeoutSec 3
   if(@($resp.data|ForEach-Object{$_.id}) -contains 'qwen3.5-4b-local'){
    $script:restored=$true
    Log 'PREACTIVE_RESTORED_MODEL_LIST_VERIFIED'
    return
   }
  }catch{}
 }
 Log 'RESTORE_NOT_VERIFIED_KEEP_WATCHDOG'
}
foreach($f in @($gate,$stdout,$stderr,$log,$receipt)){
 if(Test-Path -LiteralPath $f){throw ('Controller output collision '+$f)}
}
try {
 $now=Get-Date
 if($now.ToString('yyyyMMdd') -ne '20261016' -or $now.DayOfWeek.ToString() -ne 'Friday' -or $now.TimeOfDay -ge [TimeSpan]'11:15:00'){throw 'Outside bounded maintenance start window'}
 if((Hash $spec) -ne $specSha){throw 'R13 spec digest changed'}
 if((Hash $corpus) -ne $corpusSha){throw 'R13 corpus digest changed'}
 if((Hash $parent) -ne $parentSha){throw 'R12 parent digest changed'}
 if(-not(Test-Path -LiteralPath $recover -PathType Leaf)){throw 'Recovery controller missing'}
 if(-not(Test-Path -LiteralPath $python -PathType Leaf)){throw 'Pinned Python missing'}
 if(Test-Path -LiteralPath $output){throw 'R13 model output already exists'}
 if((@(OwnedR13)).Count){throw 'R13 trainer already active'}
 $other=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -eq 'python.exe' -and $_.CommandLine -and $_.CommandLine -match 'train_v10r2_dev|preflight_r13_behavior_launch'})
 if($other.Count){throw 'Other training process already active'}
 $head=(& git -C $repo rev-parse HEAD | Out-String).Trim()
 if($LASTEXITCODE -ne 0 -or $head -ne $ExpectedHead){throw 'Source head drift'}
 $dirty=(& git -C $repo status --porcelain=v1 | Out-String).Trim()
 if($LASTEXITCODE -ne 0 -or $dirty){throw 'Dirty source tree is not accepted'}
 foreach($name in $names){
  if((Get-ScheduledTask -TaskName $name -ErrorAction Stop).State -ne 'Running'){throw ('PreActive baseline not running '+$name)}
 }
 $live=Invoke-RestMethod 'http://127.0.0.1:18081/v1/models' -TimeoutSec 6
 if(-not(@($live.data|ForEach-Object{$_.id}) -contains 'qwen3.5-4b-local')){throw 'PreActive model baseline not healthy'}
 if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue){throw 'Recovery task collision'}
 $free=(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
 if($LASTEXITCODE -ne 0){throw 'GPU readback unavailable'}
 $os=Get-CimInstance Win32_OperatingSystem
 if(($os.FreePhysicalMemory/1048576) -lt 9 -or ($os.FreeVirtualMemory/1048576) -lt 12){throw 'Host preflight hold'}
 if($PreflightOnly){
  Write-Output ('R13_MAINTENANCE_PREFLIGHT_PASS source='+$head+' before_pause_gpu='+($free|Out-String).Trim())
  exit 0
 }
 Log ('MAINTENANCE_BEGIN head='+$head+' initial_gpu_free_mib='+($free|Out-String).Trim())
 $restoreAt=(Get-Date).AddMinutes(125)
 $trigger=New-ScheduledTaskTrigger -Once -At $restoreAt
 $action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+$recover+'" -Emergency')
 Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -User 'SYSTEM' -RunLevel Highest -Force -ErrorAction Stop | Out-Null
 $watchRegistered=$true
 $watch=Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
 if(-not $watch.Settings.Enabled){throw 'Recovery watchdog disabled'}
 Log ('RECOVERY_WATCHDOG_REGISTERED at='+$restoreAt.ToString('o'))
 $paused=$true
 foreach($name in $names){
  Stop-ScheduledTask -TaskName $name -ErrorAction Stop
  Log ('PREACTIVE_TEMPORARILY_PAUSED '+$name)
 }
 for($i=0;$i -lt 60;$i++){
  Start-Sleep -Seconds 1
  $states=@($names|ForEach-Object{(Get-ScheduledTask -TaskName $_ -ErrorAction Stop).State})
  $listener=Get-NetTCPConnection -LocalPort 18081 -State Listen -ErrorAction SilentlyContinue
  if(@($states|Where-Object{$_ -eq 'Running'}).Count -eq 0 -and -not $listener){break}
 }
 $states=@($names|ForEach-Object{(Get-ScheduledTask -TaskName $_ -ErrorAction Stop).State})
 if(@($states|Where-Object{$_ -eq 'Running'}).Count){throw 'PreActive scheduler failed to pause'}
 if(Get-NetTCPConnection -LocalPort 18081 -State Listen -ErrorAction SilentlyContinue){throw 'PreActive endpoint still active'}
 Start-Sleep -Seconds 4
 $os=Get-CimInstance Win32_OperatingSystem
 $ram=$os.FreePhysicalMemory/1048576
 $virt=$os.FreeVirtualMemory/1048576
 $gpu=(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Out-String).Trim()
 if($LASTEXITCODE -ne 0){throw 'GPU readback after pause unavailable'}
 Log ('RESOURCES_AFTER_PAUSE free_gpu_mib='+$gpu+' ram_gib='+[math]::Round($ram,2)+' virt_gib='+[math]::Round($virt,2))
 if([int]$gpu -lt 3900 -or $ram -lt 9 -or $virt -lt 12){throw 'Resource gate failed after pause'}
 $env:PYTHONPATH=$repo
 $env:PYTHONDONTWRITEBYTECODE='1'
 $args=@('-B','-u','-m','successor.experiments.preflight_r13_behavior_retry2_launch','--train','--receipt',$gate)
 $process=Start-Process -FilePath $python -WorkingDirectory $repo -ArgumentList $args -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr -ErrorAction Stop
 $trainerPid=$process.Id
 $stateResult='TRAINING_RUNNING'
 Log ('R13_TRAINER_STARTED pid='+$trainerPid)
 $deadline=(Get-Date).AddMinutes(112)
 $lastLog=(Get-Date).AddMinutes(-2)
 $consecutiveLow=0
 while(-not $process.HasExited){
  Start-Sleep -Seconds 12
  $process.Refresh()
  $os=Get-CimInstance Win32_OperatingSystem
  $ram=$os.FreePhysicalMemory/1048576
  $virt=$os.FreeVirtualMemory/1048576
  if($ram -lt 1.5 -or $virt -lt 12){$consecutiveLow++}else{$consecutiveLow=0}
  if($consecutiveLow -ge 2){
   $stateResult='STOPPED_LOW_HOST_MEMORY'
   StopOwnedR13 ('ram='+$ram+' virt='+$virt)
   break
  }
  if((Get-Date) -gt $deadline -or (Get-Date).TimeOfDay -ge [TimeSpan]'13:20:00'){
   $stateResult='STOPPED_MAINTENANCE_DEADLINE'
   StopOwnedR13 'runtime cutoff'
   break
  }
  if(((Get-Date)-$lastLog).TotalSeconds -ge 90){
   Log ('R13_TRAINING_ACTIVE free_ram='+[math]::Round($ram,2)+' free_virtual='+[math]::Round($virt,2))
   $lastLog=Get-Date
  }
 }
 $process.Refresh()
 if($process.HasExited){$trainerExited=$true;$trainerExit=[int]$process.ExitCode}
 if($stateResult.StartsWith('STOPPED_')){throw $stateResult}
 if(-not $trainerExited -or $trainerExit -ne 0){throw ('R13 trainer exited unsuccessfully='+$trainerExit)}
 $completed=Join-Path $output 'DEV_TRAINING_COMPLETE.json'
 $adapter=Join-Path $output 'adapter\adapter_model.safetensors'
 if(-not(Test-Path $completed) -or -not(Test-Path $adapter)){throw 'Adapter/receipt missing despite trainer exit 0'}
 $result=Get-Content $completed -Raw|ConvertFrom-Json
 $actualHash=Hash $adapter
 if($result.status -ne 'DEVELOPMENT_OPTIMIZER_STEP_COMPLETE' -or
   $result.spec_sha256 -ne $specSha -or $result.train_sha256 -ne $corpusSha -or
   [int]$result.cumulative_optimizer_steps -ne 84 -or
   [int]$result.resume_adapter.source_cumulative_optimizer_steps -ne 52 -or
   $result.adapter_precision_final.tensor_dtype -ne 'float32' -or
   $result.deployment_status -ne 'NOT_DEPLOYED' -or
   $actualHash -eq $parentSha){
    throw 'Final adapter or exact lineage did not qualify'
 }
 $stateResult='R13_SAVED_84_STEPS_MECHANICALLY_VERIFIED_BEHAVIOR_UNQUALIFIED'
 Log ('R13_ARTIFACT_VERIFIED sha256='+$actualHash)
} catch {
 $reason=$_.Exception.Message
 if($stateResult -eq 'TRAINING_RUNNING'){$stateResult='R13_FAILED_OR_INTERRUPTED_NOT_VERIFIED'}
 if($stateResult -eq 'NOT_STARTED'){$stateResult='R13_PREFLIGHT_HOLD_NOT_TRAINED'}
 if(Test-Path $log){Log ('R13_CONTROLLER_HOLD_OR_FAILURE '+$reason)}
 else{Write-Output ('R13_CONTROLLER_HOLD '+$reason)}
} finally {
 if($paused){
  # Ensure the exact R13 subprocess is not left holding the GPU.
  $active=@(OwnedR13)
  if($active.Count){
   try{StopOwnedR13 'controller finalization restore priority'}catch{
    Log ('FINAL_EXACT_STOP_ERROR '+$_.Exception.Message)
   }
  }
  RestorePreActive
 }
 if(-not $PreflightOnly){
 $receiptData=[ordered]@{
  schema='VERA_R13_BOUNDED_MAINTENANCE_CONTROLLER_V1'
  status=$stateResult
  reason=$reason
  source_head=$ExpectedHead
  source_spec_sha256=$specSha
  source_corpus_sha256=$corpusSha
  parent_r12_sha256=$parentSha
  target_cumulative_optimizer_steps=84
  trainer_started_pid=$trainerPid
  trainer_exited=$trainerExited
  trainer_exit_code=$trainerExit
  preactive_temporarily_paused=$paused
  preactive_restored_verified=$restored
  recovery_watchdog_task=$taskName
  behavioral_qualified=$false
  model_deployed=$false
  start_local=$start.ToString('o')
  finished_local=[DateTimeOffset]::Now.ToString('o')
 }
 try{
  [IO.File]::WriteAllText($receipt,($receiptData|ConvertTo-Json -Depth 8)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
  if(Test-Path $log){Log ('R13_CONTROLLER_FINISHED status='+$stateResult+' restored='+$restored)}
 } catch{
  Write-Output ('R13_CONTROLLER_RECEIPT_ERROR '+$_.Exception.Message)
 }
 if($watchRegistered -and ($restored -or -not $paused)){
  try{
   Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
   Log 'EMERGENCY_WATCHDOG_DISARMED_AFTER_VERIFIED_RESTORE'
  }catch{Log ('WATCHDOG_LEFT_FOR_SAFE_RECOVERY '+$_.Exception.Message)}
 }
 }
}
if($stateResult -eq 'R13_SAVED_84_STEPS_MECHANICALLY_VERIFIED_BEHAVIOR_UNQUALIFIED' -and $restored){exit 0}
exit 2
