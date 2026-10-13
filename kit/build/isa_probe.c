// Executes one instruction from the named ISA extension. Used only by build-kit.sh test
// to prove QEMU's CPU model traps instructions the emulated CPU lacks; without that,
// "the penryn build ran under -cpu Penryn" would prove nothing.
#include <string.h>

int main(int argc, char **argv) {
    const char *f = argc > 1 ? argv[1] : "";
    unsigned int a = 0, b = 1;
    unsigned long c = 3, d = 0;
    if (!strcmp(f, "ssse3")) __asm__ volatile("pshufb %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
    else if (!strcmp(f, "sse41")) __asm__ volatile("pmaxsd %%xmm1, %%xmm0" ::: "xmm0");
    else if (!strcmp(f, "sse42")) __asm__ volatile("crc32l %1, %0" : "+r"(a) : "r"(b));
    else if (!strcmp(f, "popcnt")) __asm__ volatile("popcnt %1, %0" : "=r"(d) : "r"(c));
    else if (!strcmp(f, "avx")) __asm__ volatile("vaddps %%ymm1, %%ymm0, %%ymm0" ::: "xmm0");
    else if (!strcmp(f, "avx2")) __asm__ volatile("vpaddd %%ymm1, %%ymm0, %%ymm0" ::: "xmm0");
    else return 2;
    return (int)(a + d) & 0;
}
