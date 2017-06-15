/*
 * GPUDirect Async latency benchmark
 *
 *
 * based on OFED libibverbs ud_pingpong test.
 * minimally changed to use MPI for bootstrapping,
 */
/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <time.h>
#include <assert.h>
#include <netdb.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gdsync.h>

#include "pingpong.h"
#include "gpu.h"
#include "test_utils.h"



#ifdef USE_PROF
#include "prof.h"
#else
struct prof { };
#define PROF(P, H) do { } while(0)
static inline int prof_init(struct prof *p, int unit_scale, int scale_factor, const char* unit_scale_str, int nbins, int merge_bins, const char *tags) {return 0;}
static inline int prof_destroy(struct prof *p) {return 0;}
static inline void prof_dump(struct prof *p) {}
static inline void prof_update(struct prof *p) {}
static inline void prof_enable(struct prof *p) {}
static inline int  prof_enabled(struct prof *p) { return 0; }
static inline void prof_disable(struct prof *p) {}
static inline void prof_reset(struct prof *p) {}
#endif
struct prof prof;
int prof_idx = 0;

//-----------------------------------------------------------------------------

#if 0
#define dbg(FMT, ARGS...)  do {} while(0)
#else
#define dbg_msg(FMT, ARGS...)   fprintf(stderr, "DBG [%s] " FMT, __FUNCTION__ ,##ARGS)
#define dbg(FMT, ARGS...)  dbg_msg("DBG:  ", FMT, ## ARGS)
#endif

#define min(A,B) ((A)<(B)?(A):(B))

#define USE_CUDA_PROFILER 1

enum {
	PINGPONG_RECV_WRID = 1,
	PINGPONG_SEND_WRID = 2,
};

static int page_size;

struct pingpong_context {
	struct ibv_context		*context;
	struct ibv_comp_channel	*channel;
	struct ibv_pd			*pd;
	struct ibv_mr			*mr;
	struct ibv_cq			*tx_cq;
	struct ibv_cq			*rx_cq;
	struct ibv_qp			*qp;
	struct gds_qp			*gds_qp;
	struct ibv_ah			*ah;
	struct ibv_port_attr	portinfo;
	void 	*buf;
	char 	*txbuf;
	char 	*rxbuf;
	char 	*rx_flag;
	int		size;
	int		calc_size;
	int		rx_depth;
	int		pending;
	int		gpu_id;
	int		kernel_duration;
	int		peersync;
	int		peersync_gpu_cq;
	int		consume_rx_cqe;
};

struct pingpong_dest {
	int lid;
	int qpn;
	int psn;
	union ibv_gid gid;
};

struct ib_connection {
    int             	lid;
    int            	 	qpn;
    int             	psn;
	char			 	gid[INET6_ADDRSTRLEN];
	unsigned 			rkey;
	unsigned long long 	vaddr;
};

struct app_data {
	int							port;
	int							ib_port;
	unsigned            		size;
	int                 		tx_depth;
	int 		    			sockfd;
	char						*servername;
	struct ib_connection		local_connection;
	struct ib_connection 		*remote_connection;
	struct ibv_device			*ib_dev;

};

static void print_ib_connection(char *conn_name, struct ib_connection *conn){

	printf("%s: LID %#04x, QPN %#06x, PSN %#06x RKey %#08x VAddr %#016Lx\n",
			conn_name, conn->lid, conn->qpn, conn->psn, conn->rkey, conn->vaddr);

}

/*
 *  tcp_server_listen
 * *******************
 *  Creates a TCP server socket  which listens for incoming connections
 */
static int tcp_server_listen(struct app_data *data){
	struct addrinfo *res, *t;
	struct addrinfo hints = {
		.ai_flags		= AI_PASSIVE,
		.ai_family		= AF_UNSPEC,
		.ai_socktype	= SOCK_STREAM
	};

	char *service;
	int sockfd = -1;
	int n, connfd;
	struct sockaddr_in sin;

	asprintf(&service, "%d", data->port);

	n = getaddrinfo(NULL, service, &hints, &res);

	sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);

	setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &n, sizeof n);

    bind(sockfd,res->ai_addr, res->ai_addrlen);

	listen(sockfd, 1);

	connfd = accept(sockfd, NULL, 0);

	freeaddrinfo(res);

	return connfd;
}

static int tcp_client_connect(struct app_data *data)
{
	struct addrinfo *res, *t;
	struct addrinfo hints = {
		.ai_family		= AF_UNSPEC,
		.ai_socktype	= SOCK_STREAM
	};

	char *service;
	int n;
	int sockfd = -1;
	struct sockaddr_in sin;

	asprintf(&service, "%d", data->port);

	getaddrinfo(data->servername, service, &hints, &res);

	for(t = res; t; t = t->ai_next){
		sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
		int r = connect(sockfd,t->ai_addr, t->ai_addrlen);
		if (r != 0) {
			fprintf(stderr, "Could not connect to server\n");
			return 0;
		}
	}

	freeaddrinfo(res);

	return sockfd;
}

static int tcp_exch_ib_connection_info(struct app_data *data){

	char msg[sizeof "0000:000000:000000:00000000:0000000000000000:" + INET6_ADDRSTRLEN+1];
	int parsed;

	struct ib_connection *local = &data->local_connection;
	char gid_str[INET6_ADDRSTRLEN+1];
	gid_str[INET6_ADDRSTRLEN+1] = '\0';
	strcpy(gid_str, local->gid);

	sprintf(msg, "%04x:%06x:%06x:%08x:%016Lx:%s",
				local->lid, local->qpn, local->psn, local->rkey, local->vaddr, gid_str);

	if(write(data->sockfd, msg, sizeof msg) != sizeof msg){
		perror("Could not send connection_details to peer");
		return -1;
	}

	if(read(data->sockfd, msg, sizeof msg) != sizeof msg){
		perror("Could not receive connection_details to peer");
		return -1;
	}

	if(!data->remote_connection){
		free(data->remote_connection);
	}

	data->remote_connection = malloc(sizeof(struct ib_connection));
    if (data->remote_connection == NULL){
		fprintf(stderr, "Could not allocate memory for remote_connection connection\n");
        return -1;
    }

	struct ib_connection *remote = data->remote_connection;

	parsed = sscanf(msg, "%x:%x:%x:%x:%Lx:%s",
						&remote->lid, &remote->qpn, &remote->psn, &remote->rkey, &remote->vaddr, &gid_str);
	strncpy(remote->gid, gid_str, INET6_ADDRSTRLEN);
	if(parsed != 6){
		fprintf(stderr, "Could not parse message from peer");
		free(data->remote_connection);
	}

	return 0;
}

static int pp_connect_ctx(struct pingpong_context *ctx, int port, int my_psn,
			  int sl, struct pingpong_dest *dest, int sgid_idx)
{
	struct ibv_ah_attr ah_attr = {
		.is_global     = 0,
		.dlid          = dest->lid,
		.sl            = sl,
		.src_path_bits = 0,
		.port_num      = port
	};
	struct ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR
	};

	if (ibv_modify_qp(ctx->qp, &attr, IBV_QP_STATE)) {
		fprintf(stderr, "Failed to modify QP to RTR\n");
		return 1;
	}

	attr.qp_state	    = IBV_QPS_RTS;
	attr.sq_psn	    = my_psn;

	if (ibv_modify_qp(ctx->qp, &attr,
			  IBV_QP_STATE              |
			  IBV_QP_SQ_PSN)) {
		fprintf(stderr, "Failed to modify QP to RTS\n");
		return 1;
	}

	if (dest->gid.global.interface_id) {
		ah_attr.is_global = 1;
		ah_attr.grh.hop_limit = 1;
		ah_attr.grh.dgid = dest->gid;
		ah_attr.grh.sgid_index = sgid_idx;
	}

	ctx->ah = ibv_create_ah(ctx->pd, &ah_attr);
	if (!ctx->ah) {
		union ibv_gid dgid;
		if (ibv_query_gid(ctx->context, port, 0, &dgid)) {
			fprintf(stderr, "Failed to query interface gid\n");
			return 1;
		}

		ah_attr.is_global = 1;
		ah_attr.grh.hop_limit = 1;
		ah_attr.grh.dgid = dgid;
		ah_attr.grh.sgid_index = 0;

		ctx->ah = ibv_create_ah(ctx->pd, &ah_attr);
		if (!ctx->ah) {
			fprintf(stderr, "Failed to create AH\n");
			return 1;
		}
	}

	return 0;
}

static inline unsigned long align_to(unsigned long val, unsigned long pow2)
{
	return (val + pow2 - 1) & ~(pow2 - 1);
}


static struct pingpong_context *pp_init_ctx(struct ibv_device *ib_dev, int size, int calc_size,
					    int rx_depth, int port,
					    int use_event,
					    int gpu_id,
                                            int peersync,
                                            int peersync_gpu_cq,
                                            int peersync_gpu_dbrec,
                                            int consume_rx_cqe,
                                            int sched_mode)
{
	struct pingpong_context *ctx;

	if (gpu_id >=0 && gpu_init(gpu_id, sched_mode)) {
		fprintf(stderr, "error in GPU init.\n");
		return NULL;
	}

	ctx = malloc(sizeof *ctx);
	if (!ctx)
		return NULL;

	ctx->size     = size;
	ctx->calc_size = calc_size;
	ctx->rx_depth = rx_depth;
	ctx->gpu_id   = gpu_id;

        size_t alloc_size = 3 * align_to(size + 40, page_size);
	if (ctx->gpu_id >= 0)
		ctx->buf = gpu_malloc(page_size, alloc_size);
	else
		ctx->buf = memalign(page_size, alloc_size);

	if (!ctx->buf) {
		fprintf(stderr, "Couldn't allocate work buf.\n");
		goto clean_ctx;
	}
        printf("ctx buf=%p\n", ctx->buf);
        ctx->rxbuf = (char*)ctx->buf;
        ctx->txbuf = (char*)ctx->buf + align_to(size + 40, page_size);
        //ctx->rx_flag = (char*)ctx->buf + 2 * align_to(size + 40, page_size);

        ctx->rx_flag =  memalign(page_size, alloc_size);
        if (!ctx->rx_flag) {
                fprintf(stderr, "Couldn't allocate rx_flag buf\n");
                goto clean_ctx;
        }

	ctx->kernel_duration = 0;
	ctx->peersync = peersync;
        ctx->peersync_gpu_cq = peersync_gpu_cq;
        ctx->consume_rx_cqe = consume_rx_cqe;

        // must be ZERO!!! for rx_flag to work...
	if (ctx->gpu_id >= 0)
		gpu_memset(ctx->buf, 0, alloc_size);
	else
		memset(ctx->buf, 0, alloc_size);

        memset(ctx->rx_flag, 0, alloc_size);
        gpu_register_host_mem(ctx->rx_flag, alloc_size);

        // pipe-cleaner
        gpu_launch_kernel(ctx->calc_size, ctx->peersync);
        gpu_launch_kernel(ctx->calc_size, ctx->peersync);
        gpu_launch_kernel(ctx->calc_size, ctx->peersync);
        CUCHECK(cuCtxSynchronize());

	ctx->context = ibv_open_device(ib_dev);
	if (!ctx->context) {
		fprintf(stderr, "Couldn't get context for %s\n",
			ibv_get_device_name(ib_dev));
		goto clean_buffer;
	}

	if (use_event) {
		ctx->channel = ibv_create_comp_channel(ctx->context);
		if (!ctx->channel) {
			fprintf(stderr, "Couldn't create completion channel\n");
			goto clean_device;
		}
	} else
		ctx->channel = NULL;

	ctx->pd = ibv_alloc_pd(ctx->context);
	if (!ctx->pd) {
		fprintf(stderr, "Couldn't allocate PD\n");
		goto clean_comp_channel;
	}

	ctx->mr = ibv_reg_mr(ctx->pd, ctx->buf, alloc_size, IBV_ACCESS_LOCAL_WRITE);
	if (!ctx->mr) {
		fprintf(stderr, "Couldn't register MR\n");
		goto clean_pd;
	}

        int gds_flags = 0;
        if (peersync_gpu_cq)
                gds_flags |= GDS_CREATE_QP_RX_CQ_ON_GPU;
        if (peersync_gpu_dbrec)
                gds_flags |= GDS_CREATE_QP_WQ_DBREC_ON_GPU;

        gds_qp_init_attr_t attr = {
                .send_cq = 0,
                .recv_cq = 0,
                .cap     = {
                        .max_send_wr  = rx_depth,
                        .max_recv_wr  = rx_depth,
                        .max_send_sge = 1,
                        .max_recv_sge = 1
                },
                .qp_type = IBV_QPT_UD,
        };

        ctx->gds_qp = gds_create_qp(ctx->pd, ctx->context, &attr, gpu_id, gds_flags);

        if (!ctx->gds_qp)  {
                fprintf(stderr, "Couldn't create QP\n");
                goto clean_mr;
	}
        ctx->qp = ctx->gds_qp->qp;
        ctx->tx_cq = ctx->gds_qp->qp->send_cq;
        ctx->rx_cq = ctx->gds_qp->qp->recv_cq;

	{
		struct ibv_qp_attr attr = {
			.qp_state        = IBV_QPS_INIT,
			.pkey_index      = 0,
			.port_num        = port,
			.qkey            = 0x11111111
		};

		if (ibv_modify_qp(ctx->qp, &attr,
				  IBV_QP_STATE              |
				  IBV_QP_PKEY_INDEX         |
				  IBV_QP_PORT               |
				  IBV_QP_QKEY)) {
			fprintf(stderr, "Failed to modify QP to INIT\n");
			goto clean_qp;
		}
	}

	return ctx;

clean_qp:
	gds_destroy_qp(ctx->gds_qp);

clean_mr:
	ibv_dereg_mr(ctx->mr);

clean_pd:
	ibv_dealloc_pd(ctx->pd);

clean_comp_channel:
	if (ctx->channel)
		ibv_destroy_comp_channel(ctx->channel);

clean_device:
	ibv_close_device(ctx->context);

clean_buffer:
	if (ctx->gpu_id >= 0)
		gpu_free(ctx->buf);
	else
		free(ctx->buf);

clean_ctx:
	if (ctx->gpu_id >= 0)
		gpu_finalize();
	free(ctx);

	return NULL;
}

int pp_close_ctx(struct pingpong_context *ctx)
{
	if (gds_destroy_qp(ctx->gds_qp)) {
		fprintf(stderr, "Couldn't destroy QP\n");
	}

	if (ibv_dereg_mr(ctx->mr)) {
		fprintf(stderr, "Couldn't deregister MR\n");
	}

	if (ibv_destroy_ah(ctx->ah)) {
		fprintf(stderr, "Couldn't destroy AH\n");
	}

	if (ibv_dealloc_pd(ctx->pd)) {
		fprintf(stderr, "Couldn't deallocate PD\n");
	}

	if (ctx->channel) {
		if (ibv_destroy_comp_channel(ctx->channel)) {
			fprintf(stderr, "Couldn't destroy completion channel\n");
		}
	}

	if (ibv_close_device(ctx->context)) {
		fprintf(stderr, "Couldn't release context\n");
	}

	if (ctx->gpu_id >= 0)
		gpu_free(ctx->buf);
	else
		free(ctx->buf);

	if (ctx->gpu_id >= 0)
		gpu_finalize();

	free(ctx);

	return 0;
}

static int pp_post_recv(struct pingpong_context *ctx, int n)
{
	struct ibv_sge list = {
		.addr	= (uintptr_t) ctx->rxbuf,
		.length = ctx->size + 40,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_recv_wr wr = {
		.wr_id	    = PINGPONG_RECV_WRID,
		.sg_list    = &list,
		.num_sge    = 1,
	};
	struct ibv_recv_wr *bad_wr;
	int i;

	for (i = 0; i < n; ++i)
		if (ibv_post_recv(ctx->qp, &wr, &bad_wr))
			break;

	return i;
}

static int pp_post_send(struct pingpong_context *ctx, uint32_t qpn)
{
        int ret = 0;
	struct ibv_sge list = {
		.addr	= (uintptr_t) ctx->txbuf,
		.length = ctx->size,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_send_wr wr = {
		.wr_id	    = PINGPONG_SEND_WRID,
		.sg_list    = &list,
		.num_sge    = 1,
		.opcode     = IBV_WR_SEND,
		.send_flags = IBV_SEND_SIGNALED,
		.wr         = {
			.ud = {
				 .ah          = ctx->ah,
				 .remote_qpn  = qpn,
				 .remote_qkey = 0x11111111
			 }
		}
	};
	struct ibv_send_wr *bad_wr;
        printf("ibv_post_send\n");
        return gds_post_send(ctx->gds_qp, &wr, &bad_wr);
}

static int pp_post_gpu_send(struct pingpong_context *ctx, uint32_t qpn)
{
	int ret = 0;
	struct ibv_sge list = {
		.addr	= (uintptr_t) ctx->txbuf,
		.length = ctx->size,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_exp_send_wr ewr = {
		.wr_id	    = PINGPONG_SEND_WRID,
		.sg_list    = &list,
		.num_sge    = 1,
		.exp_opcode = IBV_EXP_WR_SEND,
		.exp_send_flags = IBV_EXP_SEND_SIGNALED,
		.wr         = {
			.ud = {
				.ah          = ctx->ah,
				.remote_qpn  = qpn,
				.remote_qkey = 0x11111111
			}
		},
		.comp_mask = 0
	};
	struct ibv_exp_send_wr *bad_ewr;
	//printf("gpu_post_send_on_stream\n");
	return gds_stream_queue_send(gpu_stream, ctx->gds_qp, &ewr, &bad_ewr);
}

static int pp_post_work(struct pingpong_context *ctx, int n_posts, int rcnt, uint32_t qpn, int is_client){
	int i, ret = 0;
	int posted_recv = 0;

	//printf("post_work posting %d\n", n_posts);

	if (n_posts <= 0)
		return 0;

	posted_recv = pp_post_recv(ctx, n_posts);
	if (posted_recv < 0) {
		fprintf(stderr,"ERROR: can't post recv (%d) n_posts=%d is_client=%d\n",
			posted_recv, n_posts, is_client);
		exit(EXIT_FAILURE);
		return 0;
	} else if (posted_recv != n_posts) {
		printf(stderr,"ERROR: couldn't post all recvs (%d posted, %d requested)\n", posted_recv, n_posts);
		if (!posted_recv)
			return 0;
	}

	PROF(&prof, prof_idx++);

	for (i = 0; i < posted_recv; ++i) {
		if (is_client) {

			ret = pp_post_gpu_send(ctx, qpn);
			if (ret) {
				fprintf(stderr,"ERROR: can't post GPU send (%d) posted_recv=%d posted_so_far=%d is_client=%d \n",
					ret, posted_recv, i, is_client);
				i = -ret;
				break;
			}

			ret = gds_stream_wait_cq(gpu_stream, &ctx->gds_qp->recv_cq, ctx->consume_rx_cqe);
			if (ret) {
				fprintf(stderr,"ERROR: error in gpu_post_poll_cq (%d)\n", ret);
				i = -ret;
				break;
			}
			if (ctx->calc_size)
				gpu_launch_kernel(ctx->calc_size, ctx->peersync);
		} else {
			ret = gds_stream_wait_cq(gpu_stream, &ctx->gds_qp->recv_cq, ctx->consume_rx_cqe);
			if (ret) {
				fprintf(stderr, "ERROR: error in gpu_post_poll_cq (%d)\n", ret);
				i = -ret;
				break;
			}
			if (ctx->calc_size)
				gpu_launch_kernel(ctx->calc_size, ctx->peersync);
			ret = pp_post_gpu_send(ctx, qpn);
			if (ret) {
				fprintf(stderr, "ERROR: can't post GPU send\n");
				i = -ret;
				break;
			}
		}
	}

	PROF(&prof, prof_idx++);

	gpu_post_release_tracking_event();
	//sleep(1);
	return i;
}

int main(int argc, char *argv[])
{
	struct ibv_device		**dev_list;
	struct ibv_device		*ib_dev;
	struct pingpong_context *ctx;
	struct pingpong_dest	my_dest;
	struct pingpong_dest	*rem_dest = NULL;
	struct timeval			rstart, start, end;
	const char              *ib_devname = NULL;
	char                    *servername = NULL;
	int						port = 18515;
	int 					ib_port = 1;
	int 					size = 1024;
	int						calc_size = 128*1024;
	int						rx_depth = 2*512;
	int						iters = 1000;
	int						use_event = 0;
	int						routs;
	int						nposted;
	int						rcnt, scnt;
	int						num_cq_events = 0;
	int						sl = 0;
	int						gidx = 0;
	char			 		gid[INET6_ADDRSTRLEN];
	int						gpu_id = 0;
	int						peersync = 1;
	int						peersync_gpu_cq = 0;
	int						peersync_gpu_dbrec = 0;
	int						warmup = 10;
	int						max_batch_len = 20;
	int						consume_rx_cqe = 0;
	int						sched_mode = CU_CTX_SCHED_AUTO;
	int						ret = 0;
	// for tcp
	int 					sockfd;


	fprintf(stdout, "libgdsync build version 0x%08x, major=%d minor=%d\n", GDS_API_VERSION, GDS_API_MAJOR_VERSION, GDS_API_MINOR_VERSION);

	int version;
	ret = gds_query_param(GDS_PARAM_VERSION, &version);
	if (ret) {
		fprintf(stderr, "error querying libgdsync version\n");
		return -1;
	}
	fprintf(stdout, "libgdsync queried version 0x%08x\n", version);
	if (!GDS_API_VERSION_COMPATIBLE(version)) {
		fprintf(stderr, "incompatible libgdsync version 0x%08x\n", version);
		return -1;
	}

    struct app_data	 	 data = {
		.port	    		= 18515,
		.ib_port    		= 1,
		.size       		= 65536,
		.tx_depth   		= 100,
		.servername 		= NULL,
		.remote_connection 	= NULL,
		.ib_dev     		= NULL

	};

	srand48(getpid() * time(NULL));

	if(argc == 2){
		servername = argv[1];
	}
	if(argc > 2){
		fprintf(stderr, "*Error* Usage: <remote_server>\n");
		return -1;
	}

	if (!servername) {
		// Server side program
		printf("pid=%d server starting\n", getpid());
	} else {
		// Cliend side
		printf("pid=%d client:%s\n", getpid(), servername);
	}
    data.servername = servername;
	data.port = port;

	page_size = sysconf(_SC_PAGESIZE);

	// init local RDMA environment
	dev_list = ibv_get_device_list(NULL);
	if (!dev_list) {
		perror("Failed to get IB devices list");
		return 1;
	}


	if (!ib_devname) {
		printf("picking 1st available device\n");
		ib_dev = *dev_list;
		if (!ib_dev) {
			fprintf(stderr, "No IB devices found\n");
			return 1;
		}
	} else {
		int i;
		for (i = 0; dev_list[i]; ++i)
			if (!strcmp(ibv_get_device_name(dev_list[i]), ib_devname))
				break;
		ib_dev = dev_list[i];
		if (!ib_dev) {
			fprintf(stderr, "IB device %s not found\n", ib_devname);
			return 1;
		}
	}

	ctx = pp_init_ctx(ib_dev, size, calc_size, rx_depth, ib_port, 0, gpu_id, peersync, peersync_gpu_cq, peersync_gpu_dbrec, consume_rx_cqe, sched_mode);
	if (!ctx)
		return 1;

	int nrecv = pp_post_recv(ctx, max_batch_len);
	if (nrecv < max_batch_len) {
		fprintf(stderr, "Couldn't post receive (%d)\n", nrecv);
		return 1;
	}

	if (pp_get_port_info(ctx->context, ib_port, &ctx->portinfo)) {
		fprintf(stderr, "Couldn't get port info\n");
		return 1;
	}
	my_dest.lid = ctx->portinfo.lid;
	my_dest.qpn = ctx->qp->qp_num;
	my_dest.psn = lrand48() & 0xffffff;
	if (ibv_query_gid(ctx->context, ib_port, gidx, &my_dest.gid)) {
		fprintf(stderr, "Could not get local gid\n");
		return 1;
	}

	inet_ntop(AF_INET6, &my_dest.gid, gid, sizeof gid);
	printf("local address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x: GID %s\n",
	 	my_dest.lid, my_dest.qpn, my_dest.psn, gid);

	// exchange client server address via TCP
	if(servername){
		// client connect
		sockfd = tcp_client_connect(&data);
	}else{
		// server listen
		sockfd = tcp_server_listen(&data);
		if (!sockfd) {
			fprintf(stderr, "Error start tcp server\n");
			return 1;
		}
	}
	strcpy(data.local_connection.gid, gid);
	data.local_connection.lid = my_dest.lid;
	data.local_connection.qpn = my_dest.qpn;
	data.local_connection.psn = my_dest.psn;
	ret = tcp_exch_ib_connection_info(&data);
	if (ret != 0) {
		fprintf(stderr, "Could not exchange connection, tcp_exch_ib_connection\n");
		return 1;
	}


    ret = inet_pton(AF_INET6, &data.remote_connection->gid, &rem_dest->gid);
	if (ret != 0) {
		fprintf(stderr, "Could not convert remote GID from text to binary\n");
		return 1;
	}
	rem_dest->lid = data.remote_connection->lid;
	rem_dest->qpn = data.remote_connection->qpn;
	rem_dest->psn = data.remote_connection->psn;

	inet_ntop(AF_INET6, &rem_dest->gid, gid, sizeof gid);

	printf("remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
		rem_dest->lid, rem_dest->qpn, rem_dest->psn, gid);

	// prepared QP
	struct ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR
	};

	if (ibv_modify_qp(ctx->qp, &attr, IBV_QP_STATE)) {
		fprintf(stderr, "Failed to modify QP to RTR\n");
		return 1;
	}

	attr.qp_state	= IBV_QPS_RTS;
	attr.sq_psn	    = my_dest.psn;

	if (ibv_modify_qp(ctx->qp, &attr, IBV_QP_STATE|IBV_QP_SQ_PSN)) {
		fprintf(stderr, "Failed to modify QP to RTS\n");
		return 1;
	}

	struct ibv_ah_attr ah_attr = {
		.is_global     = 0,
		.dlid          = rem_dest->lid,
		.sl            = sl,
		.src_path_bits = 0,
		.port_num      = ib_port
	};

	ctx->ah = ibv_create_ah(ctx->pd, &ah_attr);
	if (!ctx->ah) {
		ah_attr.is_global = 1;
		ah_attr.grh.hop_limit = 1;
		ah_attr.grh.dgid = rem_dest->gid;
		ah_attr.grh.sgid_index = 0;

		ctx->ah = ibv_create_ah(ctx->pd, &ah_attr);
		if (!ctx->ah) {
			fprintf(stderr, "Failed to create AH\n");
			return 1;
		}
	}

	if (gettimeofday(&start, NULL)) {
		perror("gettimeofday");
		ret = 1;
		goto out;
	}

	// for performance reasons, multiple batches back-to-back are posted here
	rcnt = scnt = 0;
	nposted = 0;
	routs = 0;
	const int n_batches = 3;
	//int prev_batch_len = 0;
	int last_batch_len = 0;
	int n_post = 0;
	int n_posted;
	int batch;

	for (batch=0; batch<n_batches; ++batch) {
		n_post = min(min(ctx->rx_depth/2, iters-nposted), max_batch_len);
		n_posted = pp_post_work(ctx, n_post, 0, rem_dest->qpn, servername?1:0);
		if (n_posted != n_post) {
			fprintf(stderr, "ERROR: Couldn't post work, got %d requested %d\n", n_posted, n_post);
			ret = 1;
			goto out;
		}
		routs += n_posted;
		nposted += n_posted;
		//prev_batch_len = last_batch_len;
		last_batch_len = n_posted;
		printf("batch %d: posted %d sequences\n",  batch, n_posted);
	}

	ctx->pending = PINGPONG_RECV_WRID;
	float pre_post_us = 0;

	if (gettimeofday(&end, NULL)) {
		perror("gettimeofday");
		ret = 1;
		goto out;
	}
	float usec = (end.tv_sec - start.tv_sec) * 1000000 +
		(end.tv_usec - start.tv_usec);
	printf("pre-posting took %.2f usec\n", usec);
	pre_post_us = usec;

	if (!servername) {
		puts("");
		printf("batch info: rx+kernel+tx %d per batch\n", n_posted); // this is the last actually
		printf("pre-posted %d sequences in %d batches\n", nposted, 2);
		printf("GPU kernel calc buf size: %d\n", ctx->calc_size);
		printf("iters=%d tx/rx_depth=%d\n", iters, ctx->rx_depth);
		printf("\n");
		printf("testing....\n");
		fflush(stdout);
	}

	if (gettimeofday(&start, NULL)) {
		perror("gettimeofday");
		return 1;
	}
	prof_enable(&prof);
	prof_idx = 0;
	int got_error = 0;
	int iter = 0;
	while ((rcnt < iters || scnt < iters) && !got_error) {
		++iter;
		PROF(&prof, prof_idx++);

		//printf("before tracking\n"); fflush(stdout);
		int ret = gpu_wait_tracking_event(1000*1000);
		if (ret == ENOMEM) {
			printf("gpu_wait_tracking_event nothing to do (%d)\n", ret);
		} else if (ret == EAGAIN) {
			printf("gpu_wait_tracking_event timout (%d), retrying\n", ret);
			prof_reset(&prof);
			continue;
		} else if (ret) {
			fprintf(stderr, "gpu_wait_tracking_event failed (%d)\n", ret);
			got_error = ret;
		}
		//gpu_infoc(20, "after tracking\n");

		PROF(&prof, prof_idx++);

		// don't call poll_cq on events which are still being polled by the GPU
		int n_rx_ev = 0;
		if (!ctx->consume_rx_cqe) {
			struct ibv_wc wc[max_batch_len];
			int ne = 0, i;

			ne = ibv_poll_cq(ctx->rx_cq, max_batch_len, wc);
			if (ne < 0) {
				fprintf(stderr, "poll RX CQ failed %d\n", ne);
				return 1;
			}
			n_rx_ev += ne;
			//if (ne) printf("ne=%d\n", ne);
			for (i = 0; i < ne; ++i) {
				if (wc[i].status != IBV_WC_SUCCESS) {
					fprintf(stderr, "Failed status %s (%d) for wr_id %d\n",
						ibv_wc_status_str(wc[i].status),
						wc[i].status, (int) wc[i].wr_id);
					return 1;
				}

				switch ((int) wc[i].wr_id) {
					case PINGPONG_RECV_WRID:
						++rcnt;
						break;
					default:
						fprintf(stderr, "Completion for unknown wr_id %d\n",
							(int) wc[i].wr_id);
							return 1;
				}
			}
		} else {
			n_rx_ev = last_batch_len;
			rcnt += last_batch_len;
		}

		PROF(&prof, prof_idx++);
		int n_tx_ev = 0;
		struct ibv_wc wc[max_batch_len];
		int ne, i;

		ne = ibv_poll_cq(ctx->tx_cq, max_batch_len, wc);
		if (ne < 0) {
			fprintf(stderr, "poll TX CQ failed %d\n", ne);
			return 1;
		}
		n_tx_ev += ne;
		for (i = 0; i < ne; ++i) {
			if (wc[i].status != IBV_WC_SUCCESS) {
				fprintf(stderr, "Failed status %s (%d) for wr_id %d\n",
					ibv_wc_status_str(wc[i].status),wc[i].status, (int) wc[i].wr_id);
					return 1;
			}

			switch ((int) wc[i].wr_id) {
				case PINGPONG_SEND_WRID:
					++scnt;
					break;
				default:
					fprintf(stderr, "Completion for unknown wr_id %d\n",
						(int) wc[i].wr_id);
					ret = 1;
					goto out;
			}
		}
		PROF(&prof, prof_idx++);
		if (1 && (n_tx_ev || n_rx_ev)) {
			//fprintf(stderr, "iter=%d n_rx_ev=%d, n_tx_ev=%d\n", iter, n_rx_ev, n_tx_ev); fflush(stdout);
		}
		if (n_tx_ev || n_rx_ev) {
			// update counters
			routs -= last_batch_len;
			//prev_batch_len = last_batch_len;
			if (n_tx_ev != last_batch_len)
				fprintf(stderr, "[%d] unexpected tx ev %d, batch len %d\n", iter, n_tx_ev, last_batch_len);
			if (n_rx_ev != last_batch_len)
				fprintf(stderr, "[%d] unexpected rx ev %d, batch len %d\n", iter, n_rx_ev, last_batch_len);
			if (nposted < iters) {
				//fprintf(stdout, "rcnt=%d scnt=%d routs=%d nposted=%d\n", rcnt, scnt, routs, nposted); fflush(stdout);
				// potentially submit new work
				n_post = min(min(ctx->rx_depth/2, iters-nposted), max_batch_len);
				int n = pp_post_work(ctx, n_post, nposted, rem_dest->qpn, servername?1:0);
				if (n != n_post) {
					fprintf(stderr, "ERROR: post_work error (%d) rcnt=%d n_post=%d routs=%d\n", n, rcnt, n_post, routs);
					return 1;
				}
				last_batch_len = n;
				routs += n;
				nposted += n;
				//fprintf(stdout, "n_post=%d n=%d\n", n_post, n);
			}
		}
		//usleep(10);
		PROF(&prof, prof_idx++);
		prof_update(&prof);
		prof_idx = 0;

		//fprintf(stdout, "%d %d\n", rcnt, scnt); fflush(stdout);

		if (got_error) {
			fprintf(stderr, "exiting for error\n");
			return 1;
		}
	}

	if (gettimeofday(&end, NULL)) {
		perror("gettimeofday");
		ret = 1;
	}

	usec = (end.tv_sec - start.tv_sec) * 1000000 +
		(end.tv_usec - start.tv_usec) + pre_post_us;
	long long bytes = (long long) size * iters * 2;

	printf("%lld bytes in %.2f seconds = %.2f Mbit/sec\n",
		bytes, usec / 1000000., bytes * 8. / usec);
	printf(" %d iters in %.2f seconds = %.2f usec/iter\n",
		iters, usec / 1000000., usec / iters);

	if (prof_enabled(&prof)) {
		printf("dumping prof\n");
		prof_dump(&prof);
	}

	//ibv_ack_cq_events(ctx->cq, num_cq_events);

	if (pp_close_ctx(ctx))
		ret = 1;

	ibv_free_device_list(dev_list);
	//free(rem_dest);

out:
	return ret;
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
