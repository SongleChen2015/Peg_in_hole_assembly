# input: moment force z delta_z
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def fuzzy_C1(m, f, z, dz):
    # moment_max = 2
    # force_max = 20
    # z_max = 50
    # detaz_max = 3
    #
    # """"""
    # # m1 = (m + 0.5) / -0.5
    # # m2 = 1 - abs((m + 0.5) / 0.5)
    # # m3 = 1 - abs(m / 0.5)
    # # m4 = 1 - abs((m - 0.5) / 0.5)
    # # m5 = (m - 0.5) / 0.5
    #
    # m1 = 1 - m / (moment_max/4.)
    # m2 = 1 - abs((m - (moment_max/4.)) / (moment_max / 4.))
    # m3 = 1 - abs((m - (2*moment_max/4.)) / (moment_max / 4.))
    # m4 = 1 - abs((m - (3*moment_max/4.)) / (moment_max / 4.))
    # m5 = (m - (3*moment_max/4.)) / (moment_max / 4.)
    #
    # """"""
    # # f1 = (f + 5) / -15.0
    # # f2 = 1 - abs((f + 10.0) / 10.0)
    # # f3 = 1 - abs(f / 5.0)
    # # f4 = 1 - abs((f - 10.0) / 10.0)
    # # f5 = (f - 5.0) / 15.0
    #
    # f1 = 1 - f / (force_max / 4.)
    # f2 = 1 - abs((f - (force_max / 4.)) / (force_max / 4.))
    # f3 = 1 - abs((f - (2 * force_max / 4.)) / (force_max / 4.))
    # f4 = 1 - abs((f - (3 * force_max / 4.)) / (force_max / 4.))
    # f5 = (f - (3 * force_max / 4.)) / (force_max / 4.)
    #
    # m1 = max(min(m1, 1.0), 0.0)
    # m2 = max(min(m2, 1.0), 0.0)
    # m3 = max(min(m3, 1.0), 0.0)
    # m4 = max(min(m4, 1.0), 0.0)
    # m5 = max(min(m5, 1.0), 0.0)
    #
    # f1 = max(min(f1, 1.0), 0.0)
    # f2 = max(min(f2, 1.0), 0.0)
    # f3 = max(min(f3, 1.0), 0.0)
    # f4 = max(min(f4, 1.0), 0.0)
    # f5 = max(min(f5, 1.0), 0.0)
    #
    # r1 = min(m1, f1)
    # r2 = min(m1, f2)
    # r3 = min(m1, f3)
    # r4 = min(m1, f4)
    # r5 = min(m1, f5)
    # r6 = min(m2, f1)
    # r7 = min(m2, f2)
    # r8 = min(m2, f3)
    # r9 = min(m2, f4)
    # r10 = min(m2, f5)
    # r11 = min(m3, f1)
    # r12 = min(m3, f2)
    # r13 = min(m3, f3)
    # r14 = min(m3, f4)
    # r15 = min(m3, f5)
    # r16 = min(m4, f1)
    # r17 = min(m4, f2)
    # r18 = min(m4, f3)
    # r19 = min(m4, f4)
    # r20 = min(m4, f5)
    # r21 = min(m5, f1)
    # r22 = min(m5, f2)
    # r23 = min(m5, f3)
    # r24 = min(m5, f4)
    # r25 = min(m5, f5)
    #
    # r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25
    # mfoutput = (0.33 * (r1 + r5 + r21 + r25) + 0.48 * (r6 + r10 + r16 + r20) + 0.69 * (r2 + r4 + r11 + r15 + r22 + r24)
    #           + 1 * (r3 + r7 + r9 + r17 + r19 + r23) + 1.44 * (r12 + r14) + 2.07 * (r8 + r18) + 3 * (r13)) / (3 * r0)
    #
    #
    # # z1 = 1 - z / 25.0
    # # z2 = 1 - abs((z - 25.0) / 25.0)
    # # z3 = 1 - abs((z - 50.0) / 25.0)
    # # z4 = 1 - abs((z - 75.0) / 25.0)
    # # z5 = (z - 75) / 25.0
    #
    # z1 = 1 - z / (z_max / 4.)
    # z2 = 1 - abs((z - (z_max / 4.)) / (z_max / 4.))
    # z3 = 1 - abs((z - (2 * z_max / 4.)) / (z_max / 4.))
    # z4 = 1 - abs((z - (3 * z_max / 4.)) / (z_max / 4.))
    # z5 = (z - (3 * z_max / 4.)) / (z_max / 4.)
    #
    # # dz1 = 1 - dz / 0.375
    # # dz2 = 1 - abs((dz - 0.375) / 0.375)
    # # dz3 = 1 - abs((dz - 0.75) / 0.375)
    # # dz4 = 1 - abs((dz - 1.125) / 0.375)
    # # dz5 = (dz - 1.125) / 0.375
    #
    # dz1 = 1 - dz / (detaz_max / 4.)
    # dz2 = 1 - abs((dz - (detaz_max / 4.)) / (detaz_max / 4.))
    # dz3 = 1 - abs((dz - (2 * detaz_max / 4.)) / (detaz_max / 4.))
    # dz4 = 1 - abs((dz - (3 * detaz_max / 4.)) / (detaz_max / 4.))
    # dz5 = (dz - (3 * detaz_max / 4.)) / (detaz_max / 4.)
    #
    # z1 = max(min(z1, 1.0), 0.0)
    # z2 = max(min(z2, 1.0), 0.0)
    # z3 = max(min(z3, 1.0), 0.0)
    # z4 = max(min(z4, 1.0), 0.0)
    # z5 = max(min(z5, 1.0), 0.0)
    # dz1 = max(min(dz1, 1.0), 0.0)
    # dz2 = max(min(dz2, 1.0), 0.0)
    # dz3 = max(min(dz3, 1.0), 0.0)
    # dz4 = max(min(dz4, 1.0), 0.0)
    # dz5 = max(min(dz5, 1.0), 0.0)
    #
    # r1 = min(z1, dz1)
    # r2 = min(z1, dz2)
    # r3 = min(z1, dz3)
    # r4 = min(z1, dz4)
    # r5 = min(z1, dz5)
    # r6 = min(z2, dz1)
    # r7 = min(z2, dz2)
    # r8 = min(z2, dz3)
    # r9 = min(z2, dz4)
    # r10 = min(z2, dz5)
    # r11 = min(z3, dz1)
    # r12 = min(z3, dz2)
    # r13 = min(z3, dz3)
    # r14 = min(z3, dz4)
    # r15 = min(z3, dz5)
    # r16 = min(z4, dz1)
    # r17 = min(z4, dz2)
    # r18 = min(z4, dz3)
    # r19 = min(z4, dz4)
    # r20 = min(z4, dz5)
    # r21 = min(z5, dz1)
    # r22 = min(z5, dz2)
    # r23 = min(z5, dz3)
    # r24 = min(z5, dz4)
    # r25 = min(z5, dz5)
    # r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25
    # zdzoutput = (0.33 * (r1 + r2) + 0.48 * (r6 + r7) + 0.69 * (r3 + r8) + 1 * (r4 + r9 + r11 + r12 + r13) +
    #           1.44 * (r5 + r10 + r14 + r18) + 2.07 * (r15 + r19 + r16 + r17 + r23) + 3 * (
    #           r20 + r21 + r22 + r24 + r25)) / (3 * r0)
    mfoutput = fuzzy_mf(f, m)
    zdzoutput = fuzzy_zdz(z, dz)
    return mfoutput, zdzoutput

#input: mfoutput, zdzoutput
# def fuzzy_C2(m, f):
#
#     m1 = 1-m/0.25
#     m2 = 1-abs((m-0.25)/0.25)
#     m3 = 1-abs((m-0.5)/0.25)
#     m4 = 1-abs((m-0.75)/0.25)
#     m5 = (m-0.75)/0.25
#
#     f1 = 1 - f / 0.25
#     f2 = 1 - abs((f - 0.25) / 0.25)
#     f3 = 1 - abs((f - 0.5) / 0.25)
#     f4 = 1 - abs((f - 0.75) / 0.25)
#     f5 = (f - 0.75) / 0.25
#
#     m1 = max(min(m1, 1.0), 0.0)
#     m2 = max(min(m2, 1.0), 0.0)
#     m3 = max(min(m3, 1.0), 0.0)
#     m4 = max(min(m4, 1.0), 0.0)
#     m5 = max(min(m5, 1.0), 0.0)
#
#     f1 = max(min(f1, 1.0), 0.0)
#     f2 = max(min(f2, 1.0), 0.0)
#     f3 = max(min(f3, 1.0), 0.0)
#     f4 = max(min(f4, 1.0), 0.0)
#     f5 = max(min(f5, 1.0), 0.0)
#
#     r1 = min(m1, f1)
#     r2 = min(m1, f2)
#     r3 = min(m1, f3)
#     r4 = min(m1, f4)
#     r5 = min(m1, f5)
#     r6 = min(m2, f1)
#     r7 = min(m2, f2)
#     r8 = min(m2, f3)
#     r9 = min(m2, f4)
#     r10 = min(m2, f5)
#     r11 = min(m3, f1)
#     r12 = min(m3, f2)
#     r13 = min(m3, f3)
#     r14 = min(m3, f4)
#     r15 = min(m3, f5)
#     r16 = min(m4, f1)
#     r17 = min(m4, f2)
#     r18 = min(m4, f3)
#     r19 = min(m4, f4)
#     r20 = min(m4, f5)
#     r21 = min(m5, f1)
#     r22 = min(m5, f2)
#     r23 = min(m5, f3)
#     r24 = min(m5, f4)
#     r25 = min(m5, f5)
#     r0 = r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+r16+r17+r18+r19+r20+r21+r22+r23+r24+r25
#     output = (0.33*(r1+r6)+0.48*(r2+r7+r11+r16)+0.69*(r3+r12+r17+r21)
#               +1*(r4+r8+r9+r13+r22)+1.44*(r5+r10+r14+r18+r23)+2.07*(r15+r19+r24)+3*(r20+r25))/(3*r0)
#     return [output]
def fuzzy_zdz(z, dz):

    secnum = 4
    dz_max = 3.
    everysecdz = dz_max / secnum
    secdz1 = everysecdz
    secdz2 = 2 * everysecdz
    secdz3 = 3 * everysecdz

    z_max = 100
    everysecz = z_max / secnum
    secz1 = everysecz
    secz2 = 2 * everysecz
    secz3 = 3 * everysecz

    z1 = 1 - z / everysecz
    z2 = 1 - abs((z - secz1) / everysecz)
    z3 = 1 - abs((z - secz2) / everysecz)
    z4 = 1 - abs((z - secz3) / everysecz)
    z5 = (z - secz3) / everysecz

    z1 = max(min(z1, 1.0), 0.0)
    z2 = max(z2, 0.0)
    z3 = max(z3, 0.0)
    z4 = max(z4, 0.0)
    z5 = max(min(z5, 1.0), 0.0)

    dz1 = 1 - dz / everysecdz
    dz2 = 1 - abs((dz - secdz1) / everysecdz)
    dz3 = 1 - abs((dz - secdz2) / everysecdz)
    dz4 = 1 - abs((dz - secdz3) / everysecdz)
    dz5 = (dz - secdz3) / everysecdz

    dz1 = max(min(dz1, 1.0), 0.0)
    dz2 = max(dz2, 0.0)
    dz3 = max(dz3, 0.0)
    dz4 = max(dz4, 0.0)
    dz5 = max(min(dz5, 1.0), 0.0)

    r1 = min(z1, dz1)
    r2 = min(z1, dz2)
    r3 = min(z1, dz3)
    r4 = min(z1, dz4)
    r5 = min(z1, dz5)
    r6 = min(z2, dz1)
    r7 = min(z2, dz2)
    r8 = min(z2, dz3)
    r9 = min(z2, dz4)
    r10 = min(z2, dz5)
    r11 = min(z3, dz1)
    r12 = min(z3, dz2)
    r13 = min(z3, dz3)
    r14 = min(z3, dz4)
    r15 = min(z3, dz5)
    r16 = min(z4, dz1)
    r17 = min(z4, dz2)
    r18 = min(z4, dz3)
    r19 = min(z4, dz4)
    r20 = min(z4, dz5)
    r21 = min(z5, dz1)
    r22 = min(z5, dz2)
    r23 = min(z5, dz3)
    r24 = min(z5, dz4)
    r25 = min(z5, dz5)
    r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25

    # Outputs are 9 sections
    ratio = math.sqrt(2)
    zdzoutput = (4.0 * (r1 + r2) + 2.0 * ratio * (r6) + 2.0 * (r3 + r7 + r11) + ratio * (r8 + r9 + r12) +
                 1.0 * (r4 + r13) + 1 / ratio * (r5 + r16) + 0.5 * (r10 + r14 + r17 + r18) + 0.5 * ratio * (r15 + r19 + r21 + r22 + r23)
                + 0.25 * (r20 + r24 + r25)) / (4.0 * r0)

    return zdzoutput

# def fuzzy_zdz(z, dz):
#
#     secnum = 4
#     dz_max = 1.5
#     everysecdz = dz_max / secnum
#     secdz1 = everysecdz
#     secdz2 = 2 * everysecdz
#     secdz3 = 3 * everysecdz
#
#     z_max = 100
#     everysecz = z_max / secnum
#     secz1 = everysecz
#     secz2 = 2 * everysecz
#     secz3 = 3 * everysecz
#
#     z1 = 1 - z / everysecz
#     z2 = 1 - abs((z - secz1) / everysecz)
#     z3 = 1 - abs((z - secz2) / everysecz)
#     z4 = 1 - abs((z - secz3) / everysecz)
#     z5 = (z - secz3) / everysecz
#
#     z1 = max(min(z1, 1.0), 0.0)
#     z2 = max(z2, 0.0)
#     z3 = max(z3, 0.0)
#     z4 = max(z4, 0.0)
#     z5 = max(min(z5, 1.0), 0.0)
#
#     dz1 = 1 - dz / everysecdz
#     dz2 = 1 - abs((dz - secdz1) / everysecdz)
#     dz3 = 1 - abs((dz - secdz2) / everysecdz)
#     dz4 = 1 - abs((dz - secdz3) / everysecdz)
#     dz5 = (dz - secdz3) / everysecdz
#
#     dz1 = max(min(dz1, 1.0), 0.0)
#     dz2 = max(dz2, 0.0)
#     dz3 = max(dz3, 0.0)
#     dz4 = max(dz4, 0.0)
#     dz5 = max(min(dz5, 1.0), 0.0)
#
#     r1 = min(z1, dz1)
#     r2 = min(z1, dz2)
#     r3 = min(z1, dz3)
#     r4 = min(z1, dz4)
#     r5 = min(z1, dz5)
#     r6 = min(z2, dz1)
#     r7 = min(z2, dz2)
#     r8 = min(z2, dz3)
#     r9 = min(z2, dz4)
#     r10 = min(z2, dz5)
#     r11 = min(z3, dz1)
#     r12 = min(z3, dz2)
#     r13 = min(z3, dz3)
#     r14 = min(z3, dz4)
#     r15 = min(z3, dz5)
#     r16 = min(z4, dz1)
#     r17 = min(z4, dz2)
#     r18 = min(z4, dz3)
#     r19 = min(z4, dz4)
#     r20 = min(z4, dz5)
#     r21 = min(z5, dz1)
#     r22 = min(z5, dz2)
#     r23 = min(z5, dz3)
#     r24 = min(z5, dz4)
#     r25 = min(z5, dz5)
#     r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25
#
#     # Outputs are 9 sections
#     ratio = math.sqrt(2)
#     zdzoutput = (4.0 * (r1 + r2) + 2.0 * ratio * (r6) + 2.0 * (r3 + r7 + r11) + ratio * (r8 + r9 + r12) +
#                  1.0 * (r4 + r13) + 1 / ratio * (r5 + r16) + 0.5 * (r10 + r14 + r17 + r18) + 0.5 * ratio * (r15 + r19 + r21 + r22 + r23)
#                 + 0.25 * (r20 + r24 + r25)) / (-4.0 * r0)
#
#     return zdzoutput

# def fuzzy_mf(m, f):
#
#     secnum = 4
#     moment_max = 2
#     everysecm = moment_max / secnum
#     secm1 = everysecm
#     secm2 = 2 * everysecm
#     secm3 = 3 * everysecm
#
#     force_max = 20
#     everysecf = force_max / secnum
#     secf1 = everysecf
#     secf2 = 2 * everysecf
#     secf3 = 3 * everysecf
#
#     """"""
#     m1 = (m - secm3) / everysecm
#     m2 = 1 - abs(m - secm3) /everysecm
#     m3 = 1 - abs(m - secm2) / everysecm
#     m4 = 1 - abs(m - secm1) / everysecm
#     m5 = 1 - (m - secm1) / everysecm
#
#     m1 = max(min(m1, 1.0), 0.0)
#     m2 = max(m2, 0.0)
#     m3 = max(m3, 0.0)
#     m4 = max(m4, 0.0)
#     m5 = max(min(m5, 1.0), 0.0)
#
#     """"""
#     f1 = (f - secf3) / everysecf
#     f2 = 1 - abs(f - secf3) / everysecf
#     f3 = 1 - abs(f - secf2) / everysecf
#     f4 = 1 - abs(f - secf1) / everysecf
#     f5 = 1 - (f - secf1) / everysecf
#
#     f1 = max(min(f1, 1.0), 0.0)
#     f2 = max(f2, 0.0)
#     f3 = max(f3, 0.0)
#     f4 = max(f4, 0.0)
#     f5 = max(min(f5, 1.0), 0.0)
#
#     #Fuzzy rules
#     r1 = min(m1, f1)
#     r2 = min(m1, f2)
#     r3 = min(m1, f3)
#     r4 = min(m1, f4)
#     r5 = min(m1, f5)
#     r6 = min(m2, f1)
#     r7 = min(m2, f2)
#     r8 = min(m2, f3)
#     r9 = min(m2, f4)
#     r10 = min(m2, f5)
#     r11 = min(m3, f1)
#     r12 = min(m3, f2)
#     r13 = min(m3, f3)
#     r14 = min(m3, f4)
#     r15 = min(m3, f5)
#     r16 = min(m4, f1)
#     r17 = min(m4, f2)
#     r18 = min(m4, f3)
#     r19 = min(m4, f4)
#     r20 = min(m4, f5)
#     r21 = min(m5, f1)
#     r22 = min(m5, f2)
#     r23 = min(m5, f3)
#     r24 = min(m5, f4)
#     r25 = min(m5, f5)
#
#     # Outputs are 9 sections
#     ratio = math.sqrt(2)
#     r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25
#     mfoutput = (4.0 * (r1 + r2 + r6) + 2.0 * ratio * (r3 + r7 + r8) + 2.0 * (r4 + r9 + r11) + ratio * (r5 + r10 + r12) +
#                  1.0 * (r13 + r16) + 1 / ratio * (r14 + r17 + r21 + r22) + 0.5 * (r15 + r18) + 0.5 * ratio * (r19 + r20 + r23)
#                 + 0.25 * (r24 + r25)) / (-4 * r0)
#
#     return mfoutput
def fuzzy_mf(f, m):

    secnum = 4
    moment_max = 5
    everysecm = moment_max / secnum
    secm1 = everysecm
    secm2 = 2 * everysecm
    secm3 = 3 * everysecm

    force_max = 100
    everysecf = force_max / secnum
    secf1 = everysecf
    secf2 = 2 * everysecf
    secf3 = 3 * everysecf

    """"""
    m1 = (m - secm3) / everysecm
    m2 = 1 - abs(m - secm3) /everysecm
    m3 = 1 - abs(m - secm2) / everysecm
    m4 = 1 - abs(m - secm1) / everysecm
    m5 = 1 - (m - secm1) / everysecm

    m1 = max(min(m1, 1.0), 0.0)
    m2 = max(m2, 0.0)
    m3 = max(m3, 0.0)
    m4 = max(m4, 0.0)
    m5 = max(min(m5, 1.0), 0.0)

    """"""
    f1 = (f - secf3) / everysecf
    f2 = 1 - abs(f - secf3) / everysecf
    f3 = 1 - abs(f - secf2) / everysecf
    f4 = 1 - abs(f - secf1) / everysecf
    f5 = 1 - (f - secf1) / everysecf

    f1 = max(min(f1, 1.0), 0.0)
    f2 = max(f2, 0.0)
    f3 = max(f3, 0.0)
    f4 = max(f4, 0.0)
    f5 = max(min(f5, 1.0), 0.0)

    #Fuzzy rules
    r1 = min(m1, f1)
    r2 = min(m1, f2)
    r3 = min(m1, f3)
    r4 = min(m1, f4)
    r5 = min(m1, f5)
    r6 = min(m2, f1)
    r7 = min(m2, f2)
    r8 = min(m2, f3)
    r9 = min(m2, f4)
    r10 = min(m2, f5)
    r11 = min(m3, f1)
    r12 = min(m3, f2)
    r13 = min(m3, f3)
    r14 = min(m3, f4)
    r15 = min(m3, f5)
    r16 = min(m4, f1)
    r17 = min(m4, f2)
    r18 = min(m4, f3)
    r19 = min(m4, f4)
    r20 = min(m4, f5)
    r21 = min(m5, f1)
    r22 = min(m5, f2)
    r23 = min(m5, f3)
    r24 = min(m5, f4)
    r25 = min(m5, f5)

    # Outputs are 9 sections
    ratio = math.sqrt(2)
    r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25
    mfoutput = (4.0 * (r1 + r2 + r6) + 2.0 * ratio * (r3 + r7 + r8) + 2.0 * (r4 + r9 + r11) + ratio * (r5 + r10 + r12) +
                 1.0 * (r13 + r16) + 1 / ratio * (r14 + r17 + r21 + r22) + 0.5 * (r15 + r18) + 0.5 * ratio * (r19 + r20 + r23)
                + 0.25 * (r24 + r25)) / (4 * r0)

    return mfoutput


def fuzzy_C2(mf, zdz):
    secnum = 4
    mf_max = 1
    everysecmf = mf_max / secnum
    secmf1 = everysecmf
    secmf2 = 2 * everysecmf
    secmf3 = 3 * everysecmf

    mf1 = 1 - mf / everysecmf
    mf2 = 1 - abs(mf - secmf1) / everysecmf
    mf3 = 1 - abs(mf - secmf2) / everysecmf
    mf4 = 1 - abs(mf - secmf3) / everysecmf
    mf5 = (mf - secmf3) / everysecmf

    zdz_max = 1
    everyseczdz = zdz_max / secnum
    seczdz1 = everyseczdz
    seczdz2 = 2 * everyseczdz
    seczdz3 = 3 * everyseczdz

    zdz1 = 1 - zdz / everyseczdz
    zdz2 = 1 - abs(zdz - seczdz1) / everyseczdz
    zdz3 = 1 - abs(zdz - seczdz2) / everyseczdz
    zdz4 = 1 - abs(zdz - seczdz3) / everyseczdz
    zdz5 = (zdz - seczdz3) / everyseczdz

    mf1 = max(min(mf1, 1.0), 0.0)
    mf2 = max(min(mf2, 1.0), 0.0)
    mf3 = max(min(mf3, 1.0), 0.0)
    mf4 = max(min(mf4, 1.0), 0.0)
    mf5 = max(min(mf5, 1.0), 0.0)

    zdz1 = max(min(zdz1, 1.0), 0.0)
    zdz2 = max(min(zdz2, 1.0), 0.0)
    zdz3 = max(min(zdz3, 1.0), 0.0)
    zdz4 = max(min(zdz4, 1.0), 0.0)
    zdz5 = max(min(zdz5, 1.0), 0.0)

    r1 = min(mf1, zdz1)
    r2 = min(mf1, zdz2)
    r3 = min(mf1, zdz3)
    r4 = min(mf1, zdz4)
    r5 = min(mf1, zdz5)
    r6 = min(mf2, zdz1)
    r7 = min(mf2, zdz2)
    r8 = min(mf2, zdz3)
    r9 = min(mf2, zdz4)
    r10 = min(mf2, zdz5)
    r11 = min(mf3, zdz1)
    r12 = min(mf3, zdz2)
    r13 = min(mf3, zdz3)
    r14 = min(mf3, zdz4)
    r15 = min(mf3, zdz5)
    r16 = min(mf4, zdz1)
    r17 = min(mf4, zdz2)
    r18 = min(mf4, zdz3)
    r19 = min(mf4, zdz4)
    r20 = min(mf4, zdz5)
    r21 = min(mf5, zdz1)
    r22 = min(mf5, zdz2)
    r23 = min(mf5, zdz3)
    r24 = min(mf5, zdz4)
    r25 = min(mf5, zdz5)

    r0 = r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+r16+r17+r18+r19+r20+r21+r22+r23+r24+r25

    output = (0.33*(r1+r6)+0.48*(r2+r7+r11+r16)+0.69*(r3+r12+r17+r21)
              +1*(r4+r8+r9+r13+r22)+1.44*(r5+r10+r14+r18+r23)+2.07*(r15+r19+r24)+3*(r20+r25))/(-3*r0)
    return output

# mf, zdz = fuzzy_C1(1.5, 20, 0.0, 0.5)
# print(mf)
# print(zdz)
# print(fuzzy_C2(mf, zdz))
