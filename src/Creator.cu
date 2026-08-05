#include "Creator.h"
#include "Logger.h"
#include "json.h"

extern Logger logger;

namespace {

enum class KernelKind {
    Cp,
    CpInt,
    Cutcp,
    Fft,
    FftInt,
    Lbm,
    Mrif,
    MrifInt,
    Mriq,
    MriqInt,
    Sgemm,
    Stencil,
    Lava,
    Hot3d,
    Nn,
    Path,
    Tzgemm,
    Unknown
};

KernelKind getKernelKind(const std::string& name) {
    static const std::unordered_map<std::string, KernelKind> kernelKinds = {
        {"cp", KernelKind::Cp},
        {"cp_int", KernelKind::CpInt},
        {"cutcp", KernelKind::Cutcp},
        {"fft", KernelKind::Fft},
        {"fft_int", KernelKind::FftInt},
        {"lbm", KernelKind::Lbm},
        {"mrif", KernelKind::Mrif},
        {"mrif_int", KernelKind::MrifInt},
        {"mriq", KernelKind::Mriq},
        {"mriq_int", KernelKind::MriqInt},
        {"sgemm", KernelKind::Sgemm},
        {"stencil", KernelKind::Stencil},
        {"lava", KernelKind::Lava},
        {"hot3d", KernelKind::Hot3d},
        {"nn", KernelKind::Nn},
        {"path", KernelKind::Path},
        {"tzgemm", KernelKind::Tzgemm}
    };

    const auto it = kernelKinds.find(name);
    return it == kernelKinds.end() ? KernelKind::Unknown : it->second;
}

enum class MixKernelKind {
    CpFft,
    CpSgemm,
    CutcpFft,
    CutcpSgemm,
    FftLbm,
    FftMriq,
    FftSgemm,
    LbmMrif,
    LbmMriq,
    LbmSgemm,
    MrifSgemm,
    MriqSgemm,
    FftStencil,
    MrifStencil,
    Hot3dLava,
    Hot3dNn,
    Hot3dPath,
    LavaNn,
    LavaPath,
    NnPath,
    Unknown
};

MixKernelKind getMixKernelKind(const std::string& name) {
    static const std::unordered_map<std::string, MixKernelKind> kernelKinds = {
        {"cp_fft", MixKernelKind::CpFft},
        {"cp_sgemm", MixKernelKind::CpSgemm},
        {"cutcp_fft", MixKernelKind::CutcpFft},
        {"cutcp_sgemm", MixKernelKind::CutcpSgemm},
        {"fft_lbm", MixKernelKind::FftLbm},
        {"fft_mriq", MixKernelKind::FftMriq},
        {"fft_sgemm", MixKernelKind::FftSgemm},
        {"lbm_mrif", MixKernelKind::LbmMrif},
        {"lbm_mriq", MixKernelKind::LbmMriq},
        {"lbm_sgemm", MixKernelKind::LbmSgemm},
        {"mrif_sgemm", MixKernelKind::MrifSgemm},
        {"mriq_sgemm", MixKernelKind::MriqSgemm},
        {"fft_stencil", MixKernelKind::FftStencil},
        {"mrif_stencil", MixKernelKind::MrifStencil},
        {"hot3d_lava", MixKernelKind::Hot3dLava},
        {"hot3d_nn", MixKernelKind::Hot3dNn},
        {"hot3d_path", MixKernelKind::Hot3dPath},
        {"lava_nn", MixKernelKind::LavaNn},
        {"lava_path", MixKernelKind::LavaPath},
        {"nn_path", MixKernelKind::NnPath}
    };

    const auto it = kernelKinds.find(name);
    return it == kernelKinds.end() ? MixKernelKind::Unknown : it->second;
}

} // namespace

unordered_map<std::string, GPTBKernel*> kernelMap;


GPTBKernel* createKernel(const std::string &name) {
    switch (getKernelKind(name)) {
        case KernelKind::Cp:
            if (kernelMap.find("cp") == kernelMap.end()) {
                // printf("[Creator] create cp kernel\n");
                kernelMap["cp"] = new GPTBKernel(
                    10, 
                    "cp",
                    "gptb_cp", 
                    new OriCPKernel(10), 
                    dim3(SM_NUM * 6, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("cp", "ori_blks"));
            } 
            return kernelMap["cp"];
            break;
        case KernelKind::CpInt:
            if (kernelMap.find("cp_int") == kernelMap.end()) {
                // printf("[Creator] create cp kernel\n");
                kernelMap["cp_int"] = new GPTBKernel(
                    10, 
                    "cp_int",
                    "gptb_cp_int", 
                    new OriCPKernel(-1),  // for int
                    dim3(SM_NUM * 6, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("cp", "ori_blks"));
            } 
            return kernelMap["cp_int"];
            break;
        case KernelKind::Cutcp:
            if (kernelMap.find("cutcp") == kernelMap.end()) {
                kernelMap["cutcp"] = new GPTBKernel(
                    11, 
                    "cutcp",
                    "gptb_cutcp", 
                    new OriCUTCPKernel(11), 
                    dim3(SM_NUM * 6, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("cutcp", "ori_blks"));
            }
            return kernelMap["cutcp"];
            break;
        case KernelKind::Fft:
            if (kernelMap.find("fft") == kernelMap.end()) {
                kernelMap["fft"] = new GPTBKernel(
                    12, 
                    "fft",
                    "gptb_fft", 
                    new OriFFTKernel(12), 
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("fft", "ori_blks"));
            }
            return kernelMap["fft"];
            break;
        case KernelKind::FftInt:
            if (kernelMap.find("fft_int") == kernelMap.end()) {
                kernelMap["fft_int"] = new GPTBKernel(
                    12, 
                    "fft_int",
                    "gptb_fft_int", 
                    new OriFFTKernel(-1),  // for int
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("fft", "ori_blks"));
            }
            return kernelMap["fft_int"];
            break;
        case KernelKind::Lbm:
            if (kernelMap.find("lbm") == kernelMap.end()) {
                kernelMap["lbm"] = new GPTBKernel(
                    16, 
                    "lbm",
                    "gptb_lbm", 
                    new OriLBMKernel(16), 
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("lbm", "ori_blks"));
            }
            return kernelMap["lbm"];
            break;
        case KernelKind::Mrif:
            if (kernelMap.find("mrif") == kernelMap.end()) {
                kernelMap["mrif"] = new GPTBKernel(
                    17, 
                    "mrif",
                    "gptb_mrif", 
                    new OriMRIFKernel(17), 
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(256, 1, 1), 
                    0, 
                    get_kernel_info("mrif", "ori_blks"));
            }
            return kernelMap["mrif"];
            break;
        case KernelKind::MrifInt:
            if (kernelMap.find("mrif_int") == kernelMap.end()) {
                kernelMap["mrif_int"] = new GPTBKernel(
                    17, 
                    "mrif",
                    "gptb_mrif_int", 
                    new OriMRIFKernel(17), 
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(256, 1, 1), 
                    0, 
                    get_kernel_info("mrif", "ori_blks"));
            }
            return kernelMap["mrif_int"];
            break;
        case KernelKind::Mriq:
            if (kernelMap.find("mriq") == kernelMap.end()) {
                kernelMap["mriq"] = new GPTBKernel(
                    18, 
                    "mriq",
                    "gptb_mriq", 
                    new OriMRIQKernel(18), 
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(256, 1, 1), 
                    0, 
                    get_kernel_info("mriq", "ori_blks"));
            }
            return kernelMap["mriq"];
            break;
        case KernelKind::MriqInt:
            if (kernelMap.find("mriq_int") == kernelMap.end()) {
                kernelMap["mriq_int"] = new GPTBKernel(
                    18, 
                    "mriq",
                    "gptb_mriq_int", 
                    new OriMRIQKernel(18), 
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(256, 1, 1), 
                    0, 
                    get_kernel_info("mriq", "ori_blks"));
            }
            return kernelMap["mriq_int"];
            break;
        case KernelKind::Sgemm:
            if (kernelMap.find("sgemm") == kernelMap.end()) {
                kernelMap["sgemm"] = new GPTBKernel(
                    19, 
                    "sgemm",
                    "gptb_sgemm", 
                    new OriSGEMMKernel(19), 
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("sgemm", "ori_blks"));
            }
            return kernelMap["sgemm"];
            break;
        case KernelKind::Stencil:
            if (kernelMap.find("stencil") == kernelMap.end()) {
                kernelMap["stencil"] = new GPTBKernel(
                    20, 
                    "stencil",
                    "gptb_stencil", 
                    new OriSTENCILKernel(20), 
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("stencil", "ori_blks"));
            }
            return kernelMap["stencil"];
            break;
        case KernelKind::Lava:
            if (kernelMap.find("lava") == kernelMap.end()) {
                kernelMap["lava"] = new GPTBKernel(
                    20, 
                    "lava",
                    "gptb_lava", 
                    new OriLAVAKernel(20), 
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    get_kernel_info("lava", "ori_blks"));
            }
            return kernelMap["lava"];
            break;
        case KernelKind::Hot3d:
            if (kernelMap.find("hot3d") == kernelMap.end()) {
                kernelMap["hot3d"] = new GPTBKernel(
                    20, 
                    "hot3d",
                    "gptb_hot3d", 
                    new OriHOT3DKernel(20), 
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(64 * 4, 1, 1), 
                    0, 
                    get_kernel_info("hot3d", "ori_blks"));
            }
            return kernelMap["hot3d"];
            break;
        case KernelKind::Nn:
            if (kernelMap.find("nn") == kernelMap.end()) {
                kernelMap["nn"] = new GPTBKernel(
                    20, 
                    "nn",
                    "gptb_nn", 
                    new OriNNKernel(20), 
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(256, 1, 1), 
                    0, 
                    get_kernel_info("nn", "ori_blks"));
            }
            return kernelMap["nn"];
            break;
        case KernelKind::Path:
            if (kernelMap.find("path") == kernelMap.end()) {
                kernelMap["path"] = new GPTBKernel(
                    20, 
                    "path",
                    "gptb_path", 
                    new OriPATHKernel(20), 
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(256, 1, 1), 
                    0, 
                    get_kernel_info("path", "ori_blks"));
            }
            return kernelMap["path"];
            break;
        case KernelKind::Tzgemm:
            if (kernelMap.find("tzgemm") == kernelMap.end()) {
                kernelMap["tzgemm"] = new GPTBKernel(
                    20, 
                    "tzgemm",
                    "gptb_tzgemm", 
                    new OriTZGEMMKernel(20, 12800, 512, 4096), 
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    getTZGEMMGridDim(128000, 512, 4096)[3]);
            }
            printf("[Creator] create tzgemm kernel, max_blks: %d\n", getTZGEMMGridDim(128000, 512, 4096)[3]);
            return kernelMap["tzgemm"];
            break;
        case KernelKind::Unknown:
            logger.ERROR("Creator: Kernel not found: " + name);
        // Legacy configuration for cutcp:
        //     return new GPTBKernel(
        //         11, 
        //         "cutcp",
        //         "gptb_cutcp", 
        //         new OriCUTCPKernel(11), 
        //         dim3(SM_NUM * 6, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         1352);
        // Legacy configuration for fft:
        //     return new GPTBKernel(
        //         12, 
        //         "fft",
        //         "gptb_fft", 
        //         new OriFFTKernel(12), 
        //         dim3(SM_NUM * 3, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         10240);
        // Legacy configuration for lbm:
        //     return new GPTBKernel(
        //         16, 
        //         "lbm",
        //         "gptb_lbm", 
        //         new OriLBMKernel(16), 
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         16384);
        // Legacy configuration for mrif:
        //     return new GPTBKernel(
        //         17, 
        //         "mrif",
        //         "gptb_mrif", 
        //         new OriMRIFKernel(17), 
        //         dim3(SM_NUM * 3, 1, 1), 
        //         dim3(256, 1, 1), 
        //         0, 
        //         1024);
        // Legacy configuration for mriq:
        //     return new GPTBKernel(
        //         18, 
        //         "mriq",
        //         "gptb_mriq", 
        //         new OriMRIQKernel(18), 
        //         dim3(SM_NUM * 4, 1, 1), 
        //         dim3(256, 1, 1), 
        //         0, 
        //         819);
        // Legacy configuration for sgemm:
        //     return new GPTBKernel(
        //         19, 
        //         "sgemm",
        //         "gptb_sgemm", 
        //         new OriSGEMMKernel(19), 
        //         dim3(SM_NUM * 4, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         774);
        // Legacy configuration for stencil:
        //     return new GPTBKernel(
        //         20, 
        //         "stencil",
        //         "gptb_stencil", 
        //         new OriSTENCILKernel(20), 
        //         dim3(SM_NUM * 3, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         1024);
    }
    return nullptr;
}

unordered_map<std::string, MixKernel* > mixKernelMap;

MixKernel* createMixKernel(const std::string &name) {
    switch (getMixKernelKind(name)) {
        case MixKernelKind::CpFft:
            // printf("[Creator] hit cp_fft kernel\n");
            if (mixKernelMap.find("cp_fft") == mixKernelMap.end()) {
                mixKernelMap["cp_fft"] = new MixKernel(
                    0, 
                    "cp_fft", 
                    createKernel("cp"),
                    createKernel("fft"),
                    dim3(SM_NUM * 2, 1, 1), // 2
                    dim3(512, 1, 1), 
                    createKernel("cp")->gptbParams.ptb_start_block_pos,
                    createKernel("cp")->gptbParams.ptb_end_block_pos, 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["cp_fft"];
        case MixKernelKind::CpSgemm:
            if (mixKernelMap.find("cp_sgemm") == mixKernelMap.end()) {
                mixKernelMap["cp_sgemm"] = new MixKernel(
                    1, 
                    "cp_sgemm", 
                    createKernel("cp"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(256, 1, 1), 
                    createKernel("cp")->gptbParams.ptb_start_block_pos,
                    createKernel("cp")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["cp_sgemm"];
        case MixKernelKind::CutcpFft:
            if (mixKernelMap.find("cutcp_fft") == mixKernelMap.end()) {
                mixKernelMap["cutcp_fft"] = new MixKernel(
                    2, 
                    "cutcp_fft", 
                    createKernel("cutcp"),
                    createKernel("fft"),
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(256, 1, 1), 
                    createKernel("cutcp")->gptbParams.ptb_start_block_pos,
                    createKernel("cutcp")->gptbParams.ptb_end_block_pos, 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["cutcp_fft"];
        case MixKernelKind::CutcpSgemm:
            if (mixKernelMap.find("cutcp_sgemm") == mixKernelMap.end()) {
                mixKernelMap["cutcp_sgemm"] = new MixKernel(
                    3, 
                    "cutcp_sgemm", 
                    createKernel("cutcp"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(256, 1, 1), 
                    createKernel("cutcp")->gptbParams.ptb_start_block_pos,
                    createKernel("cutcp")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["cutcp_sgemm"];
        case MixKernelKind::FftLbm:
            if (mixKernelMap.find("fft_lbm") == mixKernelMap.end()) {
                mixKernelMap["fft_lbm"] = new MixKernel(
                    4, 
                    "fft_lbm", 
                    createKernel("fft"),
                    createKernel("lbm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(896, 1, 1), 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos, 
                    createKernel("lbm")->gptbParams.ptb_start_block_pos,
                    createKernel("lbm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["fft_lbm"];
        case MixKernelKind::FftMriq:
            if (mixKernelMap.find("fft_mriq") == mixKernelMap.end()) {
                mixKernelMap["fft_mriq"] = new MixKernel(
                    5, 
                    "fft_mriq", 
                    createKernel("fft"),
                    createKernel("mriq"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(896, 1, 1), 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos, 
                    createKernel("mriq")->gptbParams.ptb_start_block_pos,
                    createKernel("mriq")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["fft_mriq"];
        case MixKernelKind::FftSgemm:
            if (mixKernelMap.find("fft_sgemm") == mixKernelMap.end()) {
                mixKernelMap["fft_sgemm"] = new MixKernel(
                    6, 
                    "fft_sgemm", 
                    createKernel("fft"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(640, 1, 1), 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["fft_sgemm"];
        case MixKernelKind::LbmMrif:
            if (mixKernelMap.find("lbm_mrif") == mixKernelMap.end()) {
                mixKernelMap["lbm_mrif"] = new MixKernel(
                    7, 
                    "lbm_mrif", 
                    createKernel("lbm"),
                    createKernel("mrif"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(896, 1, 1), 
                    createKernel("lbm")->gptbParams.ptb_start_block_pos,
                    createKernel("lbm")->gptbParams.ptb_end_block_pos, 
                    createKernel("mrif")->gptbParams.ptb_start_block_pos,
                    createKernel("mrif")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["lbm_mrif"];
        case MixKernelKind::LbmMriq:
            if (mixKernelMap.find("lbm_mriq") == mixKernelMap.end()) {
                mixKernelMap["lbm_mriq"] = new MixKernel(
                    8, 
                    "lbm_mriq", 
                    createKernel("lbm"),
                    createKernel("mriq"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(640, 1, 1), 
                    createKernel("lbm")->gptbParams.ptb_start_block_pos,
                    createKernel("lbm")->gptbParams.ptb_end_block_pos, 
                    createKernel("mriq")->gptbParams.ptb_start_block_pos,
                    createKernel("mriq")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["lbm_mriq"];
        case MixKernelKind::LbmSgemm:
            if (mixKernelMap.find("lbm_sgemm") == mixKernelMap.end()) {
                mixKernelMap["lbm_sgemm"] = new MixKernel(
                    9, 
                    "lbm_sgemm", 
                    createKernel("lbm"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("lbm")->gptbParams.ptb_start_block_pos,
                    createKernel("lbm")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["lbm_sgemm"];
        case MixKernelKind::MrifSgemm:
            if (mixKernelMap.find("mrif_sgemm") == mixKernelMap.end()) {
                mixKernelMap["mrif_sgemm"] = new MixKernel(
                    10, 
                    "mrif_sgemm", 
                    createKernel("mrif"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(768, 1, 1), 
                    createKernel("mrif")->gptbParams.ptb_start_block_pos,
                    createKernel("mrif")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["mrif_sgemm"];
        case MixKernelKind::MriqSgemm:
            if (mixKernelMap.find("mriq_sgemm") == mixKernelMap.end()) {
                mixKernelMap["mriq_sgemm"] = new MixKernel(
                    11, 
                    "mriq_sgemm", 
                    createKernel("mriq"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(768, 1, 1), 
                    createKernel("mriq")->gptbParams.ptb_start_block_pos,
                    createKernel("mriq")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["mriq_sgemm"];
        case MixKernelKind::FftStencil:
            if (mixKernelMap.find("fft_stencil") == mixKernelMap.end()) {
                mixKernelMap["fft_stencil"] = new MixKernel(
                    12, 
                    "fft_stencil", 
                    createKernel("fft"),
                    createKernel("stencil"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos, 
                    createKernel("stencil")->gptbParams.ptb_start_block_pos,
                    createKernel("stencil")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["fft_stencil"];
        case MixKernelKind::MrifStencil:
            if (mixKernelMap.find("mrif_stencil") == mixKernelMap.end()) {
                mixKernelMap["mrif_stencil"] = new MixKernel(
                    13, 
                    "mrif_stencil", 
                    createKernel("mrif"),
                    createKernel("stencil"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("mrif")->gptbParams.ptb_start_block_pos,
                    createKernel("mrif")->gptbParams.ptb_end_block_pos, 
                    createKernel("stencil")->gptbParams.ptb_start_block_pos,
                    createKernel("stencil")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["mrif_stencil"];
        case MixKernelKind::Hot3dLava:
            if (mixKernelMap.find("hot3d_lava") == mixKernelMap.end()) {
                mixKernelMap["hot3d_lava"] = new MixKernel(
                    14, 
                    "hot3d_lava", 
                    createKernel("hot3d"),
                    createKernel("lava"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(384, 1, 1), 
                    createKernel("hot3d")->gptbParams.ptb_start_block_pos,
                    createKernel("hot3d")->gptbParams.ptb_end_block_pos, 
                    createKernel("lava")->gptbParams.ptb_start_block_pos,
                    createKernel("lava")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["hot3d_lava"];
        case MixKernelKind::Hot3dNn:
            if (mixKernelMap.find("hot3d_nn") == mixKernelMap.end()) {
                mixKernelMap["hot3d_nn"] = new MixKernel(
                    15, 
                    "hot3d_nn", 
                    createKernel("hot3d"),
                    createKernel("nn"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(384, 1, 1), 
                    createKernel("hot3d")->gptbParams.ptb_start_block_pos,
                    createKernel("hot3d")->gptbParams.ptb_end_block_pos, 
                    createKernel("nn")->gptbParams.ptb_start_block_pos,
                    createKernel("nn")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["hot3d_nn"];
        case MixKernelKind::Hot3dPath:
            if (mixKernelMap.find("hot3d_path") == mixKernelMap.end()) {
                mixKernelMap["hot3d_path"] = new MixKernel(
                    16, 
                    "hot3d_path", 
                    createKernel("hot3d"),
                    createKernel("path"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(512, 1, 1), 
                    createKernel("hot3d")->gptbParams.ptb_start_block_pos,
                    createKernel("hot3d")->gptbParams.ptb_end_block_pos, 
                    createKernel("path")->gptbParams.ptb_start_block_pos,
                    createKernel("path")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["hot3d_path"];
        case MixKernelKind::LavaNn:
            if (mixKernelMap.find("lava_nn") == mixKernelMap.end()) {
                mixKernelMap["lava_nn"] = new MixKernel(
                    17, 
                    "lava_nn", 
                    createKernel("lava"),
                    createKernel("nn"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(256, 1, 1), 
                    createKernel("lava")->gptbParams.ptb_start_block_pos,
                    createKernel("lava")->gptbParams.ptb_end_block_pos, 
                    createKernel("nn")->gptbParams.ptb_start_block_pos,
                    createKernel("nn")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["lava_nn"];
        case MixKernelKind::LavaPath:
            if (mixKernelMap.find("lava_path") == mixKernelMap.end()) {
                mixKernelMap["lava_path"] = new MixKernel(
                    18, 
                    "lava_path", 
                    createKernel("lava"),
                    createKernel("path"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(384, 1, 1), 
                    createKernel("lava")->gptbParams.ptb_start_block_pos,
                    createKernel("lava")->gptbParams.ptb_end_block_pos, 
                    createKernel("path")->gptbParams.ptb_start_block_pos,
                    createKernel("path")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["lava_path"];
        case MixKernelKind::NnPath:
            if (mixKernelMap.find("nn_path") == mixKernelMap.end()) {
                mixKernelMap["nn_path"] = new MixKernel(
                    19, 
                    "nn_path", 
                    createKernel("nn"),
                    createKernel("path"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(384, 1, 1), 
                    createKernel("nn")->gptbParams.ptb_start_block_pos,
                    createKernel("nn")->gptbParams.ptb_end_block_pos, 
                    createKernel("path")->gptbParams.ptb_start_block_pos,
                    createKernel("path")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["nn_path"];
        case MixKernelKind::Unknown:
            logger.ERROR("Creator: Kernel not found: " + name);
        // Legacy configuration for cp_fft:
        //     return new MixKernel(
        //         0, 
        //         "cp_fft", 
        //         (cp),
        //         (fft),
        //         dim3(SM_NUM * 2, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("cp")->gptbParams.ptb_start_block_pos,
        //         createKernel("cp")->gptbParams.ptb_end_block_pos, 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for cp_sgemm:
        //     return new MixKernel(
        //         1, 
        //         "cp_sgemm", 
        //         (cp),
        //         (sgemm),
        //         dim3(SM_NUM * 4, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("cp")->gptbParams.ptb_start_block_pos,
        //         createKernel("cp")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for cutcp_fft:
        //     return new MixKernel(
        //         2, 
        //         "cutcp_fft", 
        //         (cutcp),
        //         (fft),
        //         dim3(SM_NUM * 3, 1, 1), 
        //         dim3(786, 1, 1), 
        //         createKernel("cutcp")->gptbParams.ptb_start_block_pos,
        //         createKernel("cutcp")->gptbParams.ptb_end_block_pos, 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for cutcp_sgemm:
        //     return new MixKernel(
        //         3, 
        //         "cutcp_sgemm", 
        //         (cutcp),
        //         (sgemm),
        //         dim3(SM_NUM * 4, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("cutcp")->gptbParams.ptb_start_block_pos,
        //         createKernel("cutcp")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for fft_lbm:
        //     return new MixKernel(
        //         4, 
        //         "fft_lbm", 
        //         (fft),
        //         (lbm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(906, 1, 1), 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos, 
        //         createKernel("lbm")->gptbParams.ptb_start_block_pos,
        //         createKernel("lbm")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for fft_mriq:
        //     return new MixKernel(
        //         5, 
        //         "fft_mriq", 
        //         (fft),
        //         (mriq),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(896, 1, 1), 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos, 
        //         createKernel("mriq")->gptbParams.ptb_start_block_pos,
        //         createKernel("mriq")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for fft_sgemm:
        //     return new MixKernel(
        //         6, 
        //         "fft_sgemm", 
        //         (fft),
        //         (sgemm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(640, 1, 1), 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for lbm_mrif:
        //     return new MixKernel(
        //         7, 
        //         "lbm_mrif", 
        //         (lbm),
        //         (mrif),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(896, 1, 1), 
        //         createKernel("lbm")->gptbParams.ptb_start_block_pos,
        //         createKernel("lbm")->gptbParams.ptb_end_block_pos, 
        //         createKernel("mrif")->gptbParams.ptb_start_block_pos,
        //         createKernel("mrif")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for lbm_mriq:
        //     return new MixKernel(
        //         8, 
        //         "lbm_mriq", 
        //         (lbm),
        //         (mriq),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(640, 1, 1), 
        //         createKernel("lbm")->gptbParams.ptb_start_block_pos,
        //         createKernel("lbm")->gptbParams.ptb_end_block_pos, 
        //         createKernel("mriq")->gptbParams.ptb_start_block_pos,
        //         createKernel("mriq")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for lbm_sgemm:
        //     return new MixKernel(
        //         9, 
        //         "lbm_sgemm", 
        //         (lbm),
        //         (sgemm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("lbm")->gptbParams.ptb_start_block_pos,
        //         createKernel("lbm")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for mrif_sgemm:
        //     return new MixKernel(
        //         10, 
        //         "mrif_sgemm", 
        //         (mrif),
        //         (sgemm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(768, 1, 1), 
        //         createKernel("mrif")->gptbParams.ptb_start_block_pos,
        //         createKernel("mrif")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for mriq_sgemm:
        //     return new MixKernel(
        //         11, 
        //         "mriq_sgemm", 
        //         (mriq),
        //         (sgemm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(768, 1, 1), 
        //         createKernel("mriq")->gptbParams.ptb_start_block_pos,
        //         createKernel("mriq")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for fft_stencil:
        //     return new MixKernel(
        //         12, 
        //         "fft_stencil", 
        //         (fft),
        //         (stencil),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos, 
        //         createKernel("stencil")->gptbParams.ptb_start_block_pos,
        //         createKernel("stencil")->gptbParams.ptb_end_block_pos);
        // Legacy configuration for mrif_stencil:
        //     return new MixKernel(
        //         13, 
        //         "mrif_stencil", 
        //         (mrif),
        //         (stencil),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("mrif")->gptbParams.ptb_start_block_pos,
        //         createKernel("mrif")->gptbParams.ptb_end_block_pos, 
        //         createKernel("stencil")->gptbParams.ptb_start_block_pos,
        //         createKernel("stencil")->gptbParams.ptb_end_block_pos);

    }
    return nullptr;
}
