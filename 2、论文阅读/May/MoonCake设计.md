MoonCake设计

（1）KV cache Reuse

conductor一组预填充和解码实例，并把一个请求发送给预填充实例，这个请求包括

**原始输入**、可以被重用的prefix KV cache的key block，分配给该请求的完整缓存的key block

然后基于key block 从远GPU内存把prefix kv cache 加载到GPU显存中并启动请求

如果没有prefix cache存在跳过这一步

这一步基于三个目标：最大化重用kv cache ，平衡不同预填充节点的工作负载，确保TTFT的SLO

（2）Incremental Prefill 

预填充节点使用prefix cache并完成剩下的token （Increment KV cache）。如果剩下的uncache的输入token较多(超过指定阈值)，则分块处理