$dir = 'C:\Users\king\CUDA\Operator\LayerNorm'
$log = Join-Path $dir 'ncu_raw.txt'
$metrics = 'gpu__time_duration.sum,dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,launch__registers_per_thread,launch__shared_mem_per_block_static,sm__warps_active.avg.pct_of_peak_sustained_active'
"ncu run as $(whoami)" | Set-Content $log

foreach($v in @('V0','V1','V2','V3')){
  "##### $v #####" | Add-Content $log
  & ncu --csv --metrics $metrics -s 1 -c 1 (Join-Path $dir ($v + '.exe')) 2>&1 | Add-Content $log
}

foreach($k in @('k_copy1','k_copy4','k_noreduce','k_tree','k_shfl')){
  "##### cost / $k #####" | Add-Content $log
  & ncu --csv --metrics $metrics -k $k -c 1 (Join-Path $dir 'cost.exe') 2>&1 | Add-Content $log
}

"DONE" | Add-Content $log