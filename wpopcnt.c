#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>


uint64_t  get_cycles () {
  uint32_t lo,hi;
  asm  volatile("rdtsc":"=a"(lo),"=d"(hi));
  return  (( uint64_t)hi<<32 | lo);
}

static inline uint32_t A073642(uint64_t n)
{
// Author: Unknown
    return __builtin_popcountll(n & 0xAAAAAAAAAAAAAAAA) +
          (__builtin_popcountll(n & 0xCCCCCCCCCCCCCCCC) << 1) +
          (__builtin_popcountll(n & 0xF0F0F0F0F0F0F0F0) << 2) +
          (__builtin_popcountll(n & 0xFF00FF00FF00FF00) << 3) +
          (__builtin_popcountll(n & 0xFFFF0000FFFF0000) << 4) +
          (__builtin_popcountll(n & 0xFFFFFFFF00000000) << 5);
}

static inline uint32_t A073642_nopopcnt(uint64_t n) {
  static const uint8_t nibbles[16][16] = {
// Author: Simon Goater
{0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6},
{0, 4, 5, 9, 6, 10, 11, 15, 7, 11, 12, 16, 13, 17, 18, 22},
{0, 8, 9, 17, 10, 18, 19, 27, 11, 19, 20, 28, 21, 29, 30, 38},
{0, 12, 13, 25, 14, 26, 27, 39, 15, 27, 28, 40, 29, 41, 42, 54},
{0, 16, 17, 33, 18, 34, 35, 51, 19, 35, 36, 52, 37, 53, 54, 70},
{0, 20, 21, 41, 22, 42, 43, 63, 23, 43, 44, 64, 45, 65, 66, 86},
{0, 24, 25, 49, 26, 50, 51, 75, 27, 51, 52, 76, 53, 77, 78, 102},
{0, 28, 29, 57, 30, 58, 59, 87, 31, 59, 60, 88, 61, 89, 90, 118},
{0, 32, 33, 65, 34, 66, 67, 99, 35, 67, 68, 100, 69, 101, 102, 134},
{0, 36, 37, 73, 38, 74, 75, 111, 39, 75, 76, 112, 77, 113, 114, 150},
{0, 40, 41, 81, 42, 82, 83, 123, 43, 83, 84, 124, 85, 125, 126, 166},
{0, 44, 45, 89, 46, 90, 91, 135, 47, 91, 92, 136, 93, 137, 138, 182},
{0, 48, 49, 97, 50, 98, 99, 147, 51, 99, 100, 148, 101, 149, 150, 198},
{0, 52, 53, 105, 54, 106, 107, 159, 55, 107, 108, 160, 109, 161, 162, 214},
{0, 56, 57, 113, 58, 114, 115, 171, 59, 115, 116, 172, 117, 173, 174, 230},
{0, 60, 61, 121, 62, 122, 123, 183, 63, 123, 124, 184, 125, 185, 186, 246}};
  uint32_t i, sumsetbitpos = 0;
  for (i=0;i<16;i++) {
    sumsetbitpos += nibbles[i][(n >> (i << 2)) & 0xf];
  }
  return sumsetbitpos;
}

static inline uint64_t weighted_popcnt_SSE_lookup(const uint64_t n) {

// Author: Peter Cordes
    size_t i = 0;

    const __m128i flat_lookup = _mm_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );
    const __m128i weighted_lookup = _mm_setr_epi8(
        0, 0, 1, 1, 2, 2, 3, 3,
        3, 3, 4, 4, 5, 5, 6, 6
    );

    const __m128i low_mask = _mm_set1_epi8(0x0f);

    // normally we'd split odd/even nibbles from 128 bits of data.
    // but we're using only a single 64-bit scalar input.
//    __m128i v = _mm_set_epi64x(n>>4, n);  // high nibbles in the high half, low nibbles in the low half of a vector.
  //                                   // movq xmm0, rdi ; movdqa ; psrlw ; punpcklqdq

    __m128i v = _mm_cvtsi64_si128(n);                 // movq
    v = _mm_unpacklo_epi8( v, _mm_srli_epi16(v,4) );  // movdqa ; psrlw ; punpcklbw
               // nibbles unpacked in order: byte 3 of v is n>>(3*4) & 0xf

    const __m128i vmasked = _mm_and_si128(v, low_mask);  // pand
    __m128i counts01 = _mm_shuffle_epi8(weighted_lookup, vmasked);  // 0 to 6 according to popcnt(v & 0xa) + 2*popcnt(v & 0xc) in each nibble
    __m128i nibble_counts = _mm_shuffle_epi8(flat_lookup, vmasked);  // SSSE3 pshufb

/*
Each nibble has a different weight, so scale each byte differently when or before summing
          (__popcnt64(n & 0xF0F0F0F0F0F0F0F0) << 2) +
          (__popcnt64(n & 0xFF00FF00FF00FF00) << 3) +
          (__popcnt64(n & 0xFFFF0000FFFF0000) << 4) +
          (__popcnt64(n & 0xFFFFFFFF00000000) << 5);
                                     ^    ^
                                     |    \ 2nd nibble counted once
                                  this nibble scaled by <<3 and <<4.
                  Or <<1 and <<2 if we factor out the <<2 common to all nibble-granularity chunks
*/

   // oh duh, these are just the integers from 15 down to 0.
   __m128i scale_factors = _mm_set_epi8(0b1111<<2, 0b1110<<2, 0b1101<<2, 0b1100<<2, 0b1011<<2, 0b1010<<2, 0b1001<<2, 0b1000<<2,
                                        0b0111<<2, 0b0110<<2, 0b0101<<2, 0b0100<<2, 0b0011<<2, 0b0010<<2, 0b0001<<2, 0b0000<<2);

   // neither input has its the MSB set in any byte, so it doesn't matter which one is the unsigned or signed input.
   __m128i weighted_nibblecounts = _mm_maddubs_epi16(nibble_counts, scale_factors);  // SSSE3 pmaddubsw
   // Even if we didn't scale by the <<2 here, max element value is 15*15 + 14*15 = 435 so wouldn't fit in a byte.
   // Unfortunately byte multiply is only available with horizontal summing of pairs.


    counts01 = _mm_sad_epu8(counts01, _mm_setzero_si128()); // psadbw  hsum unsigned bytes into qword halves

    weighted_nibblecounts = _mm_madd_epi16(weighted_nibblecounts, _mm_set1_epi32(0x00010001));  // pmaddwd sum pairs of words to 32-bit dwords.  1 uop but worse latency than shift/add

    // sum the dword elements and counts01
    __m128i high = _mm_shuffle_epi32(weighted_nibblecounts, _MM_SHUFFLE(3,3, 1,1));  // pshufd or movhlps

    weighted_nibblecounts = _mm_add_epi32(weighted_nibblecounts, counts01);  // add to the bottom dword of each qword, in parallel with pshufd latency
    weighted_nibblecounts = _mm_add_epi32(weighted_nibblecounts, high);

    high = _mm_shuffle_epi32(weighted_nibblecounts, _MM_SHUFFLE(3,2, 3,2));  // pshufd, movhlps, or punpckhqdq
    weighted_nibblecounts = _mm_add_epi32(weighted_nibblecounts, high);

    return _mm_cvtsi128_si32(weighted_nibblecounts);  // movd  extract the low nibble


   // A different nibble interleave order, like 15,0, 14,1 etc. would make the max sum 225 (15*15), allowing psadbw for hsumming if we factor out the <<2
   // weighted_nibblecounts = _mm_sad_epu8(weighted_nibblecounts, _mm_setzero_si128());  // psadbw
}

static inline unsigned A073642_bithack0(uint64_t n)
{
// Author: Peter Cordes
    uint64_t ones = (n >> 1) & 0x5555555555555555;   // 0xaa>>1
    uint64_t pairs = n - ones;  // standard popcount within pairs, not weighted

    uint64_t weighted_pairs = ((ones + (ones>>2)) & 0x3333333333333333)
        + ((pairs >> 2) & 0x3333333333333333) * 2;   // 0..6 range within 4-bit nibbles
             // aka  (pairs>>1) & (0x3333333333333333<<1).  But x86-64 can efficiently use LEA to shift-and-add in one insn so it's better to shift to match the same mask.
    uint64_t quads = (pairs & 0x3333333333333333) + ((pairs >> 2) & 0x3333333333333333);   // standard popcount within each nibble, 0..4
     
    // reduce weighted pairs (0xaa and 0xcc masks) and add __popcnt64(n & 0xF0F0F0F0F0F0F0F0) << 2)
    // resulting in 8 buckets of weighted sums, each a byte wide
    uint64_t weighted_quads = ((weighted_pairs + (weighted_pairs >> 4)) & 0x0F0F0F0F0F0F0F0F) 
              + (4*((quads >> 4) & 0x0F0F0F0F0F0F0F0F));  // 0 to 2*6 + 4*4 = 28 value-range
                // need some masking before adding.  4*quads can be up to 16 so can't use (quads >> (4-2)) & 0x0F0F0F0F0F0F0F0F
    uint64_t octs = (quads + (quads >> 4)) & 0x0F0F0F0F0F0F0F0F;  //  0..8   quad fields don't have their high bit set, so we can defer masking until after adding without overflow into next field
    uint64_t hexes = (octs + (octs >> 8))  & 0x00FF00FF00FF00FF;  //  0..16 value-range

    uint64_t weighted_octs = (weighted_quads + (weighted_quads >> 8)
              + 8*((octs >> 8) )) & 0x00FF00FF00FF00FF; 
    uint64_t weighted_hexes = (weighted_octs + (weighted_octs >> 16)
                     + 16*((hexes >> 16) )); // & 0x0000FFFF0000FFFF; // 0 to 2*120 + 16*16 = 496
            // value-range 0 to 2016 fits in a 16-bit integer, so masking wider than that can be deferred
    uint64_t x32 = (hexes + (hexes >> 16)); // & 0x0000FFFF0000FFFF;  //  0..32 value-range
    return ((uint32_t)(weighted_hexes>>32) + (uint32_t)weighted_hexes + (x32>>(32-5))) & 0xFFFF;
}

static inline unsigned A073642_bithack1(uint64_t n)
{
    uint64_t ones = (n >> 1) & 0x5555555555555555;   // 0xaa>>1
    uint64_t pairs = n - ones;  // standard popcount within pairs, not weighted

    uint64_t weighted_pairs = ((ones + (ones>>2)) & 0x3333333333333333)
        + ((pairs >> 2) & 0x3333333333333333) * 2;   // 0..6 range within 4-bit nibbles
             // aka  (pairs>>1) & (0x3333333333333333<<1).  But x86-64 can efficiently use LEA to shift-and-add in one insn so it's better to shift to match the same mask.
    uint64_t quads = (pairs & 0x3333333333333333) + ((pairs >> 2) & 0x3333333333333333);   // standard popcount within each nibble, 0..4
     
    // reduce weighted pairs (0xaa and 0xcc masks) and add __popcnt64(n & 0xF0F0F0F0F0F0F0F0) << 2)
    // resulting in 8 buckets of weighted sums, each a byte wide
    uint64_t weighted_quads = ((weighted_pairs + (weighted_pairs >> 4)) & 0x0F0F0F0F0F0F0F0F) 
              + (4*((quads >> 4) & 0x0F0F0F0F0F0F0F0F));  // 0 to 2*6 + 4*4 = 28 value-range
                // need some masking before adding.  4*quads can be up to 16 so can't use (quads >> (4-2)) & 0x0F0F0F0F0F0F0F0F
    uint64_t octs = (quads + (quads >> 4)) & 0x0F0F0F0F0F0F0F0F;  //  0..8   quad fields don't have their high bit set, so we can defer masking until after adding without overflow into next field
    uint64_t hexes = (octs + (octs >> 8))  & 0x00FF00FF00FF00FF;  //  0..16 value-range

    unsigned sum_weighted_quads = (weighted_quads * 0x0101010101010101uLL)>>(64-8); // barely fits in 1 byte: max val 28 * 8 = 224
    // have to be kept separate to not overflow a byte, or would have to use weighted_octs and a 16-bit hsum
    unsigned octweight = (octs * 0x0001000100010001uLL) >> (64-8); // sum the high byte of each 16-bit group
    // can't be folded in to weighted hex sum because we need flat weights across 16-bit groups.

    // full final scale factors (*32 and *16) baked in to the multiplier; no later scaling needed
    unsigned weighted_sum_hexes = (hexes * 0x00000001000200030uLL) >> (64-16);
    return weighted_sum_hexes + sum_weighted_quads + 8*octweight;
}

static inline unsigned A073642_bithack2(uint64_t n)
{
// Author: Peter Cordes
    uint64_t ones = (n >> 1) & 0x5555555555555555;   // 0xaa>>1
    uint64_t pairs = n - ones;  // standard popcount within pairs, not weighted

    uint64_t weighted_pairs = ((ones + (ones>>2)) & 0x3333333333333333)
        + ((pairs >> 2) & 0x3333333333333333) * 2;   // 0..6 range within 4-bit nibbles
             // aka  (pairs>>1) & (0x3333333333333333<<1).  But x86-64 can efficiently use LEA to shift-and-add in one insn so it's better to shift to match the same mask.
    uint64_t quads = (pairs & 0x3333333333333333) + ((pairs >> 2) & 0x3333333333333333);   // standard popcount within each nibble, 0..4
     
    // reduce weighted pairs (0xaa and 0xcc masks) and add __popcnt64(n & 0xF0F0F0F0F0F0F0F0) << 2)
    // resulting in 8 buckets of weighted sums, each a byte wide
    uint64_t weighted_quads = ((weighted_pairs + (weighted_pairs >> 4)) & 0x0F0F0F0F0F0F0F0F) 
              + (4*((quads >> 4) & 0x0F0F0F0F0F0F0F0F));  // 0 to 2*6 + 4*4 = 28 value-range
                // need some masking before adding.  4*quads can be up to 16 so can't use (quads >> (4-2)) & 0x0F0F0F0F0F0F0F0F
    uint64_t octs = (quads + (quads >> 4)) & 0x0F0F0F0F0F0F0F0F;  //  0..8   quad fields don't have their high bit set, so we can defer masking until after adding without overflow into next field
    uint64_t hexes = (octs + (octs >> 8))  & 0x00FF00FF00FF00FF;  //  0..16 value-range

    uint64_t weighted_octs = (weighted_quads + (weighted_quads >> 8)
              + 8*((octs >> 8) )) & 0x00FF00FF00FF00FF; 
    uint64_t weighted_hexes = weighted_octs + ((hexes>>(16-4))&0x0000FFFF0000FFFF);
    unsigned sum_hexweight = (weighted_hexes * 0x0001000100010001)>>48;
    uint64_t x32_high = (hexes + (hexes<<16)) >> (48-5);  // only high half is needed
    return sum_hexweight + x32_high;
}

int main(int argc, char **argv)
{
  uint64_t ntemp, n = atol(argv[1]);
  uint32_t i, j, jtemp, k, sum, bitindex;
  uint64_t cycles1, cycles2;
  uint64_t wpop, iter;
  iter = 100000000;
  ntemp = n;
  wpop = 0;
  cycles1 = get_cycles ();
  for (i=0; i<iter; i++) {
    wpop += A073642(ntemp);
    ntemp++;
  }
  cycles2 = get_cycles ();
  printf("%li - Cycles = %li\n", wpop, cycles2 - cycles1);
  ntemp = n;
  wpop = 0;
  cycles1 = get_cycles ();
  for (i=0; i<iter; i++) {
    wpop += A073642_nopopcnt(ntemp);
    ntemp++;
  }
  cycles2 = get_cycles ();
  printf("%li - Cycles = %li\n", wpop, cycles2 - cycles1);
  ntemp = n;
  wpop = 0;
  cycles1 = get_cycles ();
  for (i=0; i<iter; i++) {
    wpop += weighted_popcnt_SSE_lookup(ntemp);
    ntemp++;
  }
  cycles2 = get_cycles ();
  printf("%li - Cycles = %li\n", wpop, cycles2 - cycles1);
  ntemp = n;
  wpop = 0;
  cycles1 = get_cycles ();
  for (i=0; i<iter; i++) {
    wpop += A073642_bithack0(ntemp);
    ntemp++;
  }
  cycles2 = get_cycles ();
  printf("%li - Cycles = %li\n", wpop, cycles2 - cycles1);
  ntemp = n;
  wpop = 0;
  cycles1 = get_cycles ();
  for (i=0; i<iter; i++) {
    wpop += A073642_bithack1(ntemp);
    ntemp++;
  }
  cycles2 = get_cycles ();
  printf("%li - Cycles = %li\n", wpop, cycles2 - cycles1);
  ntemp = n;
  wpop = 0;
  cycles1 = get_cycles ();
  for (i=0; i<iter; i++) {
    wpop += A073642_bithack2(ntemp);
    ntemp++;
  }
  cycles2 = get_cycles ();
  printf("%li - Cycles = %li\n", wpop, cycles2 - cycles1);
  exit(0);
  /*
  printf("{");
  for (i=0;i<16;i++) {
    bitindex = i << 2;
    printf("{");
    for (j=0;j<16;j++) {
      jtemp = j;
      sum = 0;
      for (k=0;k<4;k++) {
        if (jtemp & 1) sum += bitindex + k;
        jtemp = jtemp >> 1;
      }
      printf("%i", sum);
      if (j < 15) printf(", ");
    }
    if (i < 15) {
      printf("},\n");
    } else {
      printf("}}\n");
    }
  }
  */
}
