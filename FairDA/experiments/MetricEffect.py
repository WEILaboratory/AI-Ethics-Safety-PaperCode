from UaEffect import UaEffect, UaEffect_Adult
from UfEffect import UfEffect, UfEffect_Adult, UfEffect_Compas
import matplotlib.pyplot as plt

K_max = 100

(
    syn_dp_new1,
    syn_dp_upd1,
    syn_dp_new2,
    syn_dp_upd2
) = UfEffect(K_max)

# (
#     adult_dp_new1,
#     adult_dp_upd1,
#     adult_dp_new2,
#     adult_dp_upd2,
#     adult_acc_new1,
#     adult_acc_upd1,
#     adult_acc_new2,
#     adult_acc_upd2,
# ) = UfEffect_Adult(K_max)

# (
#     compas_dp_new1,
#     compas_dp_upd1,
#     compas_dp_new2,
#     compas_dp_upd2,
#     compas_acc_new1,
#     compas_acc_upd1,
#     compas_acc_new2,
#     compas_acc_upd2,
# ) = UfEffect_Compas(K_max)

# count
syn_success_ua = 0
syn_success_uf = 0
adult_success_ua = 0
adult_success_uf = 0
compas_success_ua = 0
compas_success_uf = 0

for kk in range(K_max):
    # if (syn_acc_new1[kk] - syn_acc_new2[kk]) * (
    #     syn_acc_upd1[kk] - syn_acc_upd2[kk]
    # ) <= 0:
    #     syn_success_ua += 1
    if (abs(syn_dp_new1[kk]) - abs(syn_dp_new2[kk])) * (
        abs(syn_dp_upd1[kk]) - abs(syn_dp_upd2[kk])
    ) >= 0:
        syn_success_uf += 1


# for kk in range(K_max):
#     if (adult_acc_new1[kk] - adult_acc_new2[kk]) * (
#         adult_acc_upd1[kk] - adult_acc_upd2[kk]
#     ) <= 0:
#         adult_success_ua += 1
#     if (abs(adult_dp_new1[kk]) - abs(adult_dp_new2[kk])) * (
#         abs(adult_dp_upd1[kk]) - abs(adult_dp_upd2[kk])
#     ) >= 0:
#         adult_success_uf += 1

# for kk in range(K_max):
#     if (compas_acc_new1[kk] - compas_acc_new2[kk]) * (
#         compas_acc_upd1[kk] - compas_acc_upd2[kk]
#     ) <= 0:
#         compas_success_ua += 1
#     if (abs(compas_dp_new1[kk]) - abs(compas_dp_new2[kk])) * (
#         abs(compas_dp_upd1[kk]) - abs(compas_dp_upd2[kk])
#     ) >= 0:
#         compas_success_uf += 1

print(syn_success_ua, syn_success_uf)
print(adult_success_ua, adult_success_uf)
print(compas_success_ua, compas_success_uf)

print("end")

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# fig.suptitle("Metric Effectiveness")

# axs[0].plot(range(K_max), dp_new1, color="blue", linestyle="dashed", label="$U_a(D_1)$")
# axs[0].plot(
#     range(K_max), dp_upd1, color="blue", label="Model's Accuracy trained on $D_1$"
# )
# axs[0].plot(
#     range(K_max),
#     dp_new2,
#     color="red",
#     linestyle="dashed",
#     label="$U_a(D_2)$",
# )
# axs[0].plot(
#     range(K_max), dp_upd2, color="red", label="Model's Accuracy trained on $D_2$"
# )
# axs[0].legend(ncol=1, loc="best")
# axs[0].set_xlabel("Number of trials")
# axs[0].set_ylabel("Accuracy")

# axs[1].plot(
#     range(K_max), dp_new1_p, color="blue", linestyle="dashed", label="$U_f(D_1)$"
# )
# axs[1].plot(range(K_max), dp_upd1_p, color="blue", label="Model's DP trained on $D_1$")
# axs[1].plot(
#     range(K_max), dp_new2_p, color="red", linestyle="dashed", label="$U_f(D_2)$"
# )
# axs[1].plot(range(K_max), dp_upd2_p, color="red", label="Model's DP trained on $D_2$")
# axs[1].legend(ncol=1, loc="best")
# axs[1].set_xlabel("Number of trials")
# axs[1].set_ylabel("DP")
# # axs[1].plot(
# #     range(K_max), dp_new1_n, color="green", linestyle="dashed", label="$U_f(D_3)$"
# # )
# # axs[1].plot(range(K_max), dp_upd1_n, color="green", label="Model's DP trained on $D_3$")
# # axs[1].plot(
# #     range(K_max), dp_new2_n, color="orange", linestyle="dashed", label="$U_f(D_4)$"
# # )
# # axs[1].plot(
# #     range(K_max), dp_upd2_n, color="orange", label="Model's DP trained on $D_4$"
# # )

# plt.savefig("fig/MetricEff.eps", format="eps")
# plt.savefig("fig/MetricEff.jpg")
