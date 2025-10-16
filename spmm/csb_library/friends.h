#ifndef _FRIENDS_H_
#define _FRIENDS_H_

#include <iostream>
#include <algorithm>
#include "bicsb.h"
#include "utility.h"

using namespace std;	

template <class NU, class IU>	
class BiCsb;

template <class NU, class IU, unsigned UUDIM>
class BmCsb;

double prescantime;


#if (__GNUC__ == 4 && (__GNUC_MINOR__ < 7) )
#define emplace_back push_back
#endif

// SpMM operation
/**
  * Operation y = A*x+y on a semiring SR
  * A: a general CSB matrix (no specialization on booleans is necessary as this loop is independent of numerical values) 
  * x: a column vector or a set of column vectors (i.e. array of structs, array of std:arrays, etc))
  * SR::multiply() handles the multiple rhs and type promotions, etc. 
 **/
template <typename SR, typename NT, typename IT, typename RHS, typename LHS>
void bicsb_gespmv(const BiCsb<NT, IT> & A, const RHS * __restrict x, LHS * __restrict y)
{
    IT ysize = A.lowrowmask + 1; // size of the output subarray (per block row - except the last)

    if (A.isPar())
    {
        float rowave = static_cast<float>(A.numnonzeros()) / (A.nbr - 1);

        #pragma omp parallel for schedule(dynamic)
        for (IT i = 0; i < A.nbr; ++i)
        {
            IT *btop = A.top[i];                  // pointer to block row
            IT rhi = ((i << A.rowlowbits) & A.highrowmask);
            LHS *suby = &y[rhi];

            IT threshold = std::max(static_cast<NT>(BALANCETH * rowave), static_cast<NT>(BREAKEVEN * ysize));

            // Large/Irregular block row -> chunking + BMult
            if (btop[A.nbc] - btop[0] > threshold)
            {
                // Build chunks for load balancing
                std::vector<IT*> chunks;
                chunks.push_back(btop);

                for (IT j = 0; j < A.nbc;)
                {
                    IT count = btop[j + 1] - btop[j];

                    if (count < BREAKEVEN * ysize && j < A.nbc)
                    {
                        while (count < BREAKEVEN * ysize && j < A.nbc)
                        {
                            count += btop[(++j) + 1] - btop[j];
                        }
                        chunks.push_back(btop + j); // push chunk boundary
                    }
                    else
                    {
                        chunks.push_back(btop + (++j));
                    }
                }

                // Call BMult on chunks
                if (i == A.nbr - 1)
                    A.template BMult<SR>(&chunks[0], 0, chunks.size() - 1, x, suby, A.rowsize() - ysize * i);
                else
                    A.template BMult<SR>(&chunks[0], 0, chunks.size() - 1, x, suby, ysize);
            }
            else
            {
                // Small / dense block: use sequential SubSpMV
                A.template SubSpMV<SR>(btop, 0, A.nbc, x, suby);
            }
        }
    }
    else
    {
        // Sequential fallback
        for (IT i = 0; i < A.nbr; ++i)
        {
            IT *btop = A.top[i];
            IT rhi = ((i << A.rowlowbits) & A.highrowmask);
            LHS *suby = &y[rhi];
            A.template SubSpMV<SR>(btop, 0, A.nbc, x, suby);
        }
    }
}


#endif

