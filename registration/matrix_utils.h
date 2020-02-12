//
// Created by dch on 17/06/16.
//

#pragma once
namespace Gadgetron {
// ----------------------------------------------------------------------------
// Numerical diagonalization of 3x3 matrcies
// Copyright (C) 2006  Joachim Kopp
// ----------------------------------------------------------------------------
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
// ----------------------------------------------------------------------------


// Constants
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)

// Macros
#define SQR(x)      ((x)*(x))                        // x^2


// ----------------------------------------------------------------------------
    __inline__ __device__ void dsyevc3(vector_td<float,3> G, vector_td<float,3> w)
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
    {
            float m, c1, c0;

            // Determine coefficients of characteristic poynomial. We write
            //       | a   d   f  |
            //  A =  | d*  b   e  |
            //       | f*  e*  c  |
            float de = G[0]*G[1] * G[1]*G[2];                                    // d * e
            float dd = SQR(G[0]*G[1]);                                         // d^2
            float ee = SQR(G[1]*G[2]);                                         // e^2
            float ff = SQR(G[0]*G[2]);                                         // f^2
            m  = G[0]*G[0] + G[1]*G[1] + G[2]*G[2];
            c1 = (G[0]*G[0]*G[1]*G[1] + G[0]*G[0]*G[2]*G[2] + G[1]*G[1]*G[2]*G[2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
                 - (dd + ee + ff);
            c0 = G[2]*G[2]*dd + G[0]*G[0]*ee + G[1]*G[1]*ff - G[0]*G[0]*G[1]*G[1]*G[2]*G[2]
                 - 2.0 * G[0]*G[2]*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

            float p, sqrt_p, q, c, s, phi;
            p = SQR(m) - 3.0*c1;
            q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
            sqrt_p = sqrt(fabs(p));

            phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
            phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);

            c = sqrt_p*cos(phi);
            s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

            w[1]  = (1.0/3.0)*(m - c);
            w[2]  = w[1] + s;
            w[0]  = w[1] + c;
            w[1] -= s;
            
    }





// ----------------------------------------------------------------------------
    __device__ void dsyevv3(const vector_td<float,3> G, vector_td<float,3>& Q[3], vector_td<float,3>& w)
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
//     (according to the documentation, only the upper triangular part needs
//     to be filled)
//   v1.0: First released version
// ----------------------------------------------------------------------------
    {

            float norm;          // Squared norm or inverse norm of current eigenvector
            float n0, n1;        // Norm of first and second columns of A
            float n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
            float thresh;        // Small number used as threshold for floating point comparisons
            float error;         // Estimated maximum roundoff error in some steps
            float wmax;          // The eigenvalue of maximum modulus
            float f, t;          // Intermediate storage
            int i, j;             // Loop counters


            // Calculate eigenvalues
            dsyevc3(G, w);


            wmax = fabs(w[0]);
            if ((t=fabs(w[1])) > wmax)
                    wmax = t;
            if ((t=fabs(w[2])) > wmax)
                    wmax = t;
            thresh = SQR(8.0 * DBL_EPSILON * wmax);

            // Prepare calculation of eigenvectors
            n0tmp   = SQR(G[0]*G[1]) + SQR(G[0]*G[2]);
            n1tmp   = SQR(G[0]*G[1]) + SQR(G[1]*G[2]);
            Q[0][1] = G[0]*G[1]*G[1]*G[2] - G[0]*G[2]*G[1]*G[1];
            Q[1][1] = G[0]*G[2]*G[0]*G[1] - G[1]*G[2]*G[0]*G[0];
            Q[2][1] = SQR(G[0]*G[1]);

            // Calculate first eigenvector by the formula
            //   v[0] = (A - w[0]).e1 x (A - w[0]).e2
            G[0]*G[0] -= w[0];
            G[1]*G[1] -= w[0];
            Q[0][0] = Q[0][1] + G[0]*G[2]*w[0];
            Q[1][0] = Q[1][1] + G[1]*G[2]*w[0];
            Q[2][0] = G[0]*G[0]*G[1]*G[1] - Q[2][1];
            norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
            n0      = n0tmp + SQR(G[0]*G[0]);
            n1      = n1tmp + SQR(G[1]*G[1]);
            error   = n0 * n1;

            if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
            {
                    Q[0][0] = 1.0;
                    Q[1][0] = 0.0;
                    Q[2][0] = 0.0;
            }
            else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
            {
                    Q[0][0] = 0.0;
                    Q[1][0] = 1.0;
                    Q[2][0] = 0.0;
            }
            else if (norm < SQR(32.0 * FLT_EPSILON) * error)
            {                         // If angle between A[0] and A[1] is too small, don't use
                    t = SQR(G[0]*G[1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
                    f = -G[0]*G[0] / G[0]*G[1];
                    if (SQR(G[1]*G[1]) > t)
                    {
                            t = SQR(G[1]*G[1]);
                            f = -G[0]*G[1] / G[1]*G[1];
                    }
                    if (SQR(G[1]*G[2]) > t)
                            f = -G[0]*G[2] / G[1]*G[2];
                    norm    = 1.0/sqrt(1 + SQR(f));
                    Q[0][0] = norm;
                    Q[1][0] = f * norm;
                    Q[2][0] = 0.0;
            }
            else                      // This is the standard branch
            {
                    norm = sqrt(1.0 / norm);
                    for (j=0; j < 3; j++)
                            Q[j][0] = Q[j][0] * norm;
            }


            // Prepare calculation of second eigenvector
            t = w[0] - w[1];

            Q[0][1]  = Q[0][1] + G[0]*G[2]*w[1];
            Q[1][1]  = Q[1][1] + G[1]*G[2]*w[1];
            Q[2][1]  = (G[0]*G[0]+t)*(G[1]*G[1]+t) - Q[2][1];
            norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
            n0       = n0tmp + SQR(G[0]*G[0]+t);
            n1       = n1tmp + SQR(G[1]*G[1]+t);
            error    = n0 * n1;

            if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
            {
                    Q[0][1] = 1.0;
                    Q[1][1] = 0.0;
                    Q[2][1] = 0.0;
            }
            else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
            {
                    Q[0][1] = 0.0;
                    Q[1][1] = 1.0;
                    Q[2][1] = 0.0;
            }
            else if (norm < SQR(64.0 * DBL_EPSILON) * error)
            {                       // If angle between A[0] and A[1] is too small, don't use
                    t = SQR(G[0]*G[1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
                    f = -G[0]*G[0] / G[0]*G[1];
                    if (SQR(G[1]*G[1]) > t)
                    {
                            t = SQR(G[1]*G[1]);
                            f = -G[0]*G[1] / G[1]*G[1];
                    }
                    if (SQR(G[1]*G[2]) > t)
                            f = -G[0]*G[2] / G[1]*G[2];
                    norm    = 1.0/sqrt(1 + SQR(f));
                    Q[0][1] = norm;
                    Q[1][1] = f * norm;
                    Q[2][1] = 0.0;
            }
            else
            {
                    norm = sqrt(1.0 / norm);
                    for (j=0; j < 3; j++)
                            Q[j][1] = Q[j][1] * norm;
            }




            // Calculate third eigenvector according to
            //   v[2] = v[0] x v[1]
            Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
            Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
            Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];



    }



};
