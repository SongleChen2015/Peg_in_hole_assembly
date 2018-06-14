import numpy as np
#input: moment force z delta_z
def fuzzy_C1(m, f, z, dz):

    m1 = (m + 0.5) / -0.5
    m2 = 1 - abs((m + 0.5) / 0.5)
    m3 = 1 - abs(m / 0.5)
    m4 = 1 - abs((m - 0.5) / 0.5)
    m5 = (m - 0.5) / 0.5

    secnum =4
    force_max = 80
    everysecf = force_max / secnum
    secf1 = everysecf
    secf2 = 2 * everysecf
    secf3 = 3 * everysecf

    # f1 = (f + 5) / -15.0
    # f2 = 1 - abs((f + 10.0) / 10.0)
    # f3 = 1 - abs(f / 5.0)
    # f4 = 1 - abs((f - 10.0) / 10.0)
    # f5 = (f - 5.0) / 15.0
    f1 = (f - secf3) / everysecf
    f2 = 1 - abs(f - secf3) / everysecf
    f3 = 1 - abs(f - secf2) / everysecf
    f4 = 1 - abs(f - secf1) / everysecf
    f5 = 1 - (f - secf1) / everysecf

    m1 = max(min(m1, 1.0), 0.0)
    m2 = max(min(m2, 1.0), 0.0)
    m3 = max(min(m3, 1.0), 0.0)
    m4 = max(min(m4, 1.0), 0.0)
    m5 = max(min(m5, 1.0), 0.0)
    f1 = max(min(f1, 1.0), 0.0)
    f2 = max(min(f2, 1.0), 0.0)
    f3 = max(min(f3, 1.0), 0.0)
    f4 = max(min(f4, 1.0), 0.0)
    f5 = max(min(f5, 1.0), 0.0)
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
    r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25
    mfoutput = (0.33 * (r1 + r5 + r21 + r25) + 0.48 * (r6 + r10 + r16 + r20) + 0.69 * (r2 + r4 + r11 + r15 + r22 + r24)
              + 1 * (r3 + r7 + r9 + r17 + r19 + r23) + 1.44 * (r12 + r14) + 2.07 * (r8 + r18) + 3 * (r13)) / (3 * r0)

    z1 = 1 - z / 25.0
    z2 = 1 - abs((z - 25.0) / 25.0)
    z3 = 1 - abs((z - 50.0) / 25.0)
    z4 = 1 - abs((z - 75.0) / 25.0)
    z5 = (z - 75) / 25.0
    dz1 = 1 - dz / 0.375
    dz2 = 1 - abs((dz - 0.375) / 0.375)
    dz3 = 1 - abs((dz - 0.75) / 0.375)
    dz4 = 1 - abs((dz - 1.125) / 0.375)
    dz5 = (dz - 1.125) / 0.375
    z1 = max(min(z1, 1.0), 0.0)
    z2 = max(min(z2, 1.0), 0.0)
    z3 = max(min(z3, 1.0), 0.0)
    z4 = max(min(z4, 1.0), 0.0)
    z5 = max(min(z5, 1.0), 0.0)
    dz1 = max(min(dz1, 1.0), 0.0)
    dz2 = max(min(dz2, 1.0), 0.0)
    dz3 = max(min(dz3, 1.0), 0.0)
    dz4 = max(min(dz4, 1.0), 0.0)
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
    zdzoutput = (0.33 * (r1 + r2) + 0.48 * (r6 + r7) + 0.69 * (r3 + r8) + 1 * (r4 + r9 + r11 + r12 + r13) +
              1.44 * (r5 + r10 + r14 + r18) + 2.07 * (r15 + r19 + r16 + r17 + r23) + 3 * (
              r20 + r21 + r22 + r24 + r25)) / (3 * r0)
    return mfoutput, zdzoutput

#input: mfoutput, zdzoutput
def fuzzy_C2(m, f):
    m1 = 1-m/0.25
    m2 = 1-abs((m-0.25)/0.25)
    m3 = 1-abs((m-0.5)/0.25)
    m4 = 1-abs((m-0.75)/0.25)
    m5 = (m-0.75)/0.25
    f1 = 1 - f / 0.25
    f2 = 1 - abs((f - 0.25) / 0.25)
    f3 = 1 - abs((f - 0.5) / 0.25)
    f4 = 1 - abs((f - 0.75) / 0.25)
    f5 = (f - 0.75) / 0.25
    m1 = max(min(m1, 1.0), 0.0)
    m2 = max(min(m2, 1.0), 0.0)
    m3 = max(min(m3, 1.0), 0.0)
    m4 = max(min(m4, 1.0), 0.0)
    m5 = max(min(m5, 1.0), 0.0)
    f1 = max(min(f1, 1.0), 0.0)
    f2 = max(min(f2, 1.0), 0.0)
    f3 = max(min(f3, 1.0), 0.0)
    f4 = max(min(f4, 1.0), 0.0)
    f5 = max(min(f5, 1.0), 0.0)
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
    r0 = r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+r16+r17+r18+r19+r20+r21+r22+r23+r24+r25
    output = (0.33*(r1)+0.48*(r2+r6+r7)+0.69*(r3+r4+r8+r11+r12+r16)
              +1*(r5+r9+r13+r17+r21)+1.44*(r10+r14+r15+r18+r22+r23)+2.07*(r19+r24+r20)+3*(r25))/(3*r0)
    return output

