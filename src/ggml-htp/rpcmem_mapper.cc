#include "rpcmem_mapper.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "dsprpc_interface.h"
#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"

void RpcMemMapper::validate(const ggml_tensor * dst) {
    std::vector<ggml_backend_buffer *> buffers;

    auto add_buffer = [&](ggml_backend_buffer * buf) {
        if (ggml_backend_buft_is_rpcmem(buf->buft)) {
            buffers.push_back(buf);
        }
    };

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src) {
            add_buffer(src->buffer);
        }
    }
    add_buffer(dst->buffer);

    size_t required_size = 0;
    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        if (buf_mapping.count(buf_base)) {
            // buffer already mapped, we put it at the front of the LRU access list
            accessed_bufs.erase(buf_iters.at(buf_base));
            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base] = accessed_bufs.begin();
        } else {
            required_size += buf->size;
        }
    }

    GGML_ASSERT(required_size <= max_active_map_size);
    while (active_map_size + required_size > max_active_map_size) {
        // remove least recent used mapping
        void * buf_base     = accessed_bufs.back();
        auto [fd, buf_size] = buf_mapping.at(buf_base);
        int err             = 0;

        if (defer_unmap) {
            // We assume slightly exceeding the planned max_active_map_size is acceptable
            pending_unmap_reqs.emplace_back(fd, buf_base, buf_size);
        } else {
            fprintf(stderr, "[1]rpcmem_mapper: removing memory mapping for rpcmem buffer %p, size %.5f MiB, fd %d\n", buf_base,
                    buf_size / 1048576.0, fd);
            err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf_base, buf_size);
            if (err) {
                fprintf(stderr, "fastrpc_munmap failed with return code: %x\n", err);
            }
            ++n_map_ops;
        }

        if (!defer_unmap && err) {
            fprintf(stderr, "CRITICAL: fastrpc_munmap failed. Keeping mapping to prevent state corruption.\n");
            break;
        }

        accessed_bufs.pop_back();
        buf_iters.erase(buf_base);
        buf_mapping.erase(buf_base);
        active_map_size -= buf_size;
    }

    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        size_t buf_size = buf->size;
        if (buf_mapping.count(buf_base)) {
            continue;
        }

        int fd = rpcmem_to_fd(buf_base);
        if (fd < 0) {
            GGML_ABORT("rpcmem_to_fd returns %d, ptr %p, for dst tensor %s, n_ops %d\n", fd, buf_base, dst->name,
                       n_map_ops);
        }

        auto it = std::find_if(pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
                               [fd](const auto & v) { return std::get<0>(v) == fd; });
        if (it == pending_unmap_reqs.end()) {
            // int retry = 0;

            // TODO(hzx): fix this and remove reference code
            // fastrpc_munmap does not release DSP's vm mapping immediately
again:
            int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf_base, 0, buf_size, FASTRPC_MAP_FD);
            if (err) {
                dump_state();

                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    auto src = dst->src[i];
                    if (!src) {
                        continue;
                    }
                    fprintf(stderr, "  src index %d name %s", i, src->name);
                    if (ggml_backend_buft_is_rpcmem(src->buffer->buft)) {
                        auto base = ggml_backend_buffer_get_base(src->buffer);
                        fprintf(stderr, " buf %p\n", base);
                    } else {
                        fprintf(stderr, " (non rpcmem)\n");
                    }
                }

                fprintf(stderr, "dst name: %s, op: %s, n buffers to map in this step: %ld, active size: %.2f MiB\n",
                        dst->name, ggml_op_name(dst->op), buffers.size(), active_map_size / 1048576.0);

                /*
                if (++retry < 3) {
                    // try: unmap all other buffers
                    std::unordered_set<void *> buf_ptrs;
                    for (auto b : buffers) {
                        buf_ptrs.insert(ggml_backend_buffer_get_base(b));
                    }

                    for (auto it = buf_mapping.begin(); it != buf_mapping.end();) {
                        auto ptr = it->first;
                        if (buf_ptrs.count(ptr)) {
                            ++it;
                            continue;  // skip needed buffers
                        }
                        auto [fd, len] = it->second;
                        auto lru_iter  = buf_iters.at(ptr);
                        accessed_bufs.erase(lru_iter);
                        buf_iters.erase(ptr);
                        it = buf_mapping.erase(it);

                        int e = fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, len);
                        fprintf(stderr, "try unmap fd %d addr %p len %ld ret %x\n", fd, ptr, len, e);

                        rpcmem_free(ptr);
                        fprintf(stderr, "danger operation: free %p\n", ptr);
                    }

                    goto again;
                }
                    */

                GGML_ABORT(
                    "fastrpc_mmap failed with return code: 0x%x fd: %d buf_base: %p buf_size: %ld buf usage: %d\n", err,
                    fd, buf_base, buf_size, buf->usage);
            }

            ++n_map_ops;
        } else {
            pending_unmap_reqs.erase(it);
        }

        fprintf(stderr, "rpcmem_mapper: creating memory mapping for rpcmem buffer %p, size %.5f MiB, fd %d\n", buf_base,
                buf_size / 1048576.0, fd);

        accessed_bufs.push_front(buf_base);
        buf_iters[buf_base]   = accessed_bufs.begin();
        buf_mapping[buf_base] = { fd, buf_size };
        active_map_size += buf_size;
    }
}

#include <atomic>
#include <cstring>
#include "message.h"
#include <dlfcn.h>
#include <unistd.h>

template <typename T> void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

static void send_dsp_unmap_request(int fd) {
    auto * ctx = ggml_backend_htp_context::instance();
    if (!ctx->ops_backend_initialized || ctx->ops_msg_chan == nullptr) {
        return;
    }

    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);
    
    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    std::atomic_store(d_ptr, (uint64_t)0);

    msg_hdr->n_reqs = 1;

    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);

    auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
    
    RequestHeader req_hdr{
        .state = 0,
        .type  = REQUEST_TYPE_RPCMEM_MAP,
    };
    write_buf(p, req_hdr);

    RpcmemMapRequest map_req{
        .n_puts = 1,
        .n_gets = 0,
    };
    write_buf(p, map_req);

    write_buf(p, fd);

    if (1) {
        uint32_t sum = 0;
        uint32_t * begin = ((uint32_t *) msg_hdr) + 3; 
        uint32_t * end = ((uint32_t *) msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;

        for (auto * cur = begin; cur < end; ++cur) {
            sum += *cur;
        }
        sum += 0x00000001; // v0=1, v1=0
        msg_hdr->checksum = -sum;
    
#ifdef __aarch64__
        asm volatile("dmb sy" ::: "memory");
#endif
    }

    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));
    
    std::atomic_store_explicit(v0_ptr, (uint8_t)1, std::memory_order_release);

    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        usleep(1); 
    }

    d_ptr->store(0, std::memory_order_relaxed);
    
    // printf("RpcMemMapper: DSP successfully executed put_map for fd %d\n", fd);
}

#include <unistd.h>

// 需要注意的是，相关buffer是由DSP端mmap_manager_get_map增加了引用计数，所以释放的时候，也需要调用mmap_manager_put_map来减少引用计数，才能真正释放。
// 否则fastrpc_munmap会报错
void RpcMemMapper::free_buffer(void * ptr) {
    if (ptr == nullptr) return;

    printf("RpcMemMapper: free_buffer called for ptr %p\n", ptr);

    auto try_munmap_with_retry = [](int fd, void * addr, size_t size, const char* tag) -> bool {
        int err = 0;
        for (int i = 0; i < 5; i++) {
            err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, addr, size);
            if (err == 0 || i == 4) {
                return true;
            }
            
            // 20000 us = 20 ms
            fprintf(stderr, "RpcMemMapper [%s]: fastrpc_munmap busy/failed (0x%x), retrying %d/5...\n", tag, err, i + 1);
            usleep(20000); 
        }
        
        fprintf(stderr, "RpcMemMapper [%s]: CRITICAL WARNING: munmap failed 0x%x after 5 retries. LEAKING buffer to prevent crash.\n", tag, err);
        return false;
    };

    bool found_in_pending = false;
    for (auto it = pending_unmap_reqs.begin(); it != pending_unmap_reqs.end(); ++it) {
        if (std::get<1>(*it) == ptr) {
            auto [fd, addr, size] = *it;
            found_in_pending = true;

            printf("[ERROR]should not be here\n");
            // 不应该运行到这里，pending_unmap_reqs，中的buffer，应该在htp_ops_compute_op函数执行的结尾就被释放（首先计数器减一，再munmap）
        }
    }

    if (!found_in_pending && buf_mapping.count(ptr)) {
        auto [fd, size] = buf_mapping.at(ptr);
        
        fprintf(stderr, "RpcMemMapper: freeing active buffer %p, fd %d\n", ptr, fd);

        send_dsp_unmap_request(fd);

        if (!try_munmap_with_retry(fd, ptr, size, "Active")) {
            return;
        }

        if (buf_iters.count(ptr)) {
            accessed_bufs.erase(buf_iters.at(ptr));
            buf_iters.erase(ptr);
        }
        buf_mapping.erase(ptr);
        
        if (active_map_size >= size) {
            active_map_size -= size;
        } else {
            active_map_size = 0;
        }
    } 

    rpcmem_free(ptr);
}

std::pair<int, ssize_t> RpcMemMapper::get_tensor_mapping(const ggml_tensor* tensor) const {
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(tensor->buffer->buft));

    void * buf_base = ggml_backend_buffer_get_base(tensor->buffer);
    auto [fd, _]    = buf_mapping.at(buf_base);
    auto offset     = (intptr_t) tensor->data - (intptr_t) buf_base;
    return { fd, offset };
}

void RpcMemMapper::unmap_all_pending_buffers() {
    for (auto it = pending_unmap_reqs.begin(); it != pending_unmap_reqs.end();) {
        auto [fd, buf_base, buf_size] = *it;

        fprintf(stderr, "[2]rpcmem_mapper: removing memory mapping for rpcmem buffer %p, size %.5f MiB, fd %d\n", buf_base,
                buf_size / 1048576.0, fd);
        int err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf_base, buf_size);
        if (err == 0) {
            ++n_map_ops;
            it = pending_unmap_reqs.erase(it);
        } else {
            fprintf(stderr, "fastrpc_munmap failed: 0x%x. Moving back to active mappings.\n", err);

            // 必须把记录还给活跃列表，防止下次 validate 触发重复 mmap
            buf_mapping[buf_base] = {fd, buf_size};
            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base] = accessed_bufs.begin();
            active_map_size += buf_size;

            it = pending_unmap_reqs.erase(it);
        }
    }
}

void RpcMemMapper::dump_state() const {
    fprintf(stderr, "total %d fastrpc_mmap + fastrpc_munmap ops\n", n_map_ops);

    if (!buf_mapping.empty()) {
        fprintf(stderr, "active mappings:\n");
    }
    for (const auto & [addr, pair] : buf_mapping) {
        fprintf(stderr, "    addr %p -> fd %d, size %.2f MiB\n", addr, pair.first, pair.second / 1048576.0);
    }

    if (!pending_unmap_reqs.empty()) {
        fprintf(stderr, "pending unmap requests:\n");
    }
    for (const auto & [fd, addr, size] : pending_unmap_reqs) {
        fprintf(stderr, "    fd %d, addr %p, size %.2f MiB\n", fd, addr, size / 1048576.0);
    }
}

extern "C" {

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst) {
    ggml_backend_htp_context::instance()->mapper.validate(dst);
    return 0;
}
}
