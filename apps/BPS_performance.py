#!/usr/bin/env python

import mokka
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))

logger = logging.getLogger("mokka.BPS_MI_performance")
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--model_file", help="Save file for model parameters", required=False
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot results", default=False
    )
    parser.add_argument(
        "--model_type", help="Which model to validate", default="reproduce"
    )
    parser.add_argument(
        "--mapper_type",
        help="Which mapper type is used",
        default="nn",
        choices=("nn", "simple", "separated", "separated_qam", "qam"),
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Debug mode", default=False
    )
    parser.add_argument(
        "--demapper", action="store_true", help="use demapper to calculate BMI"
    )
    parser.add_argument("--demapper_type", default=None, help="demapper type to use")
    parser.add_argument(
        "--random_seed",
        default=None,
        help="Random seed to seed random number generator",
        type=int,
    )
    parser.add_argument("-o", "--output_file", default=None, help="Output file")
    parser.add_argument(
        "--validation_lw_steps",
        default=10,
        help="Number of validation steps across the linewidth range",
        type=int,
    )
    parser.add_argument(
        "--validation_snr_steps",
        default=10,
        help="Number of validation steps acress the SNR range",
        type=int,
    )
    parser.add_argument(
        "--validation_size",
        default=10000,
        help="Number of validation samples per batch",
        type=int,
    )
    parser.add_argument(
        "--validation_count",
        default=1000,
        help="Number of validation rounds per point",
        type=int,
    )
    parser.add_argument("--snr_min", default=15, help="Minimum SNR (in dB)", type=float)
    parser.add_argument("--snr_max", default=25, help="Maximum SNR (in dB)", type=float)
    parser.add_argument(
        "--lw_min", default=50, help="Minimum linewidth (in kHz)", type=float
    )
    parser.add_argument(
        "--lw_max", default=600, help="Maximum linewidth (in kHz)", type=float
    )
    parser.add_argument(
        "--estimation",
        default=None,
        choices=(None, "under", "over"),
        help="Estimation error ",
    )
    parser.add_argument("--bps-file", default=None, help="File for trainable BPS")
    return parser.parse_args()


def main():  # noqa
    args = parse_args()
    if args.debug:
        mokka.utils.setup_logging(logging.DEBUG)
    else:
        mokka.utils.setup_logging(logging.INFO)
    m = 6
    sym_rate = 32e9  # symbol rate [Hz]
    T_sym = 1 / sym_rate  # symbol duration [s]
    min_delta_f = args.lw_min * 1e3
    max_delta_f = args.lw_max * 1e3
    validation_lw_steps = args.validation_lw_steps
    validation_snr_steps = args.validation_snr_steps
    min_snr = args.snr_min
    max_snr = args.snr_max
    validation_size = args.validation_size
    validation_count = args.validation_count
    val_delta_f = torch.linspace(min_delta_f, max_delta_f, validation_lw_steps)[:, None]
    val_sigma_phi = np.sqrt(2 * np.pi * val_delta_f * T_sym)
    val_snr = torch.linspace(min_snr, max_snr, validation_snr_steps)[:, None]
    val_N0_over = mokka.utils.N0(val_snr + 2)
    val_N0_under = mokka.utils.N0(val_snr - 2)
    # Blind Phase Search prop/valerties
    Mtestangles = 60  # 4 * 36  # Number of test angles: 36
    window_size_BPS = 128  # Number of samples used as sliding windows: 40
    no_sections = 1
    start_phase_width = 2 * np.pi
    if args.mapper_type == "qam":
        Mtestangles = int(Mtestangles // 4)
        no_sections = 4
        start_phase_width = 0
    # Additive White Gaussian Noise (AWGN) properties
    SNR_lin = 10 ** (val_snr / 10)
    val_sigma_n = torch.sqrt(1 / (SNR_lin))
    val_N0 = val_sigma_n**2

    pilot = False
    pilot_interval = 32

    if args.random_seed is not None:
        torch.random.manual_seed(args.random_seed)

    # Only use BPS & Phasenoise Wiener for validation/evaluation
    PN_Wiener = mokka.channels.torch.PhasenoiseWiener(
        start_phase_width=start_phase_width
    )
    if args.mapper_type == "nn":
        mapper = mokka.mapping.torch.ConstellationMapper.load_model(
            torch.load(args.model_file, map_location=device)
        )
    elif args.mapper_type == "simple":
        mapper = mokka.mapping.torch.SimpleConstellationMapper.load_model(
            torch.load(args.model_file, map_location=device)
        )
    elif args.mapper_type == "separated":
        mapper = mokka.mapping.torch.SeparatedConstellationMapper.load_model(
            torch.load(args.model_file, map_location=device)
        )
    elif args.mapper_type == "separated_qam":
        mapper = mokka.mapping.torch.SeparatedConstellationMapper.load_model(
            torch.load(args.model_file, map_location=device)
        )
    elif args.mapper_type == "qam":
        mapper = mokka.mapping.torch.QAMConstellationMapper(m)
    model = mapper.to(device)
    if args.bps_file:
        BPS = mokka.synchronizers.phase.torch.BPS(
            Mtestangles,
            None,
            window_size_BPS,
            True,
            torch.tensor(0.001),
            no_sections,
            trainable=True,
            avg_filter="rect"
        ).to(device)
        bps_weights = torch.load(args.bps_file, map_location=device)
        BPS.load_state_dict(bps_weights)
    else:
        BPS = mokka.synchronizers.phase.torch.BPS(
            Mtestangles,
            mapper.get_constellation(),
            window_size_BPS,
            False,
            0,
            no_sections,
            avg_filter_type="rect"
        ).to(device)

    if args.demapper:
        if args.demapper_type == "mismatched":
            demapper = mokka.mapping.torch.ClassicalDemapper(
                torch.tensor(0.001),
                model.get_constellation(),
                optimize=True,
            ).to(device)
            demapper.update_constellation(model.get_constellation())
        elif args.demapper_type == "separated":
            demapper = mokka.mapping.torch.SeparatedSimpleDemapper.load_model(
                torch.load(args.model_file, map_location=device)
            ).to(device)

        else:
            demapper = mokka.mapping.torch.ConstellationDemapper.load_model(
                torch.load(args.model_file, map_location=device)
            ).to(device)
    # Generate validation dataset
    # Create validation set
    validation_bits = torch.tensor(
        mokka.utils.generators.numpy.generate_bits((validation_size, model.m.item())),
        device=device,
        dtype=torch.float32,
    )
    validation_data = (
        val_N0.unsqueeze(1)
        .repeat(1, val_sigma_phi.size()[0], 1)
        .flatten()[:, None]
        .to(device),
        val_sigma_phi.unsqueeze(0)
        .repeat(val_N0.size()[0], 1, 1)
        .flatten()[:, None]
        .to(device),
    )
    validation_data_under = (
        val_N0_under.unsqueeze(1)
        .repeat(1, val_sigma_phi.size()[0], 1)
        .flatten()[:, None]
        .to(device),
        val_sigma_phi.unsqueeze(0)
        .repeat(val_N0.size()[0], 1, 1)
        .flatten()[:, None]
        .to(device),
    )
    validation_data_over = (
        val_N0_over.unsqueeze(1)
        .repeat(1, val_sigma_phi.size()[0], 1)
        .flatten()[:, None]
        .to(device),
        val_sigma_phi.unsqueeze(0)
        .repeat(val_N0.size()[0], 1, 1)
        .flatten()[:, None]
        .to(device),
    )
    validation_N0 = validation_data[0].expand(-1, validation_size)
    validation_lw = validation_data[1].expand(-1, validation_size)
    validation_set = (validation_bits, validation_N0, validation_lw)
    validation_set_over = (
        validation_bits,
        validation_data_over[0].expand(-1, validation_size),
        validation_lw,
    )
    validation_set_under = (
        validation_bits,
        validation_data_under[0].expand(-1, validation_size),
        validation_lw,
    )

    validation_idx = np.arange(validation_set[0].shape[0], dtype=int)

    # with torch.no_grad():
    with torch.enable_grad():
        # model.eval()
        # if args.demapper:
        #     demapper.eval()
        # Run validation with non-diff (hard) BPS
        all_val_bmis = []
        for v in torch.arange(validation_lw_steps * validation_snr_steps):
            logger.info(
                "validation step %s/%s",
                (v + 1).item(),
                validation_lw_steps * validation_snr_steps,
            )
            val_bmi_loop = []

            for val_id in range(validation_count):
                logger.info("validation count: %s/%s", val_id + 1, validation_count)
                if args.demapper == "mismatched":
                    demapper.sigma_n = torch.tensor(0.001)
                val_idx = np.random.choice(
                    validation_idx,
                    size=validation_size,  # int(batch_size_per_epoch[e])
                )
                sim_args = tuple(t[v, val_idx][:, None] for t in validation_set[1:])
                sim_args_under = tuple(
                    t[v, val_idx][:, None] for t in validation_set_under[1:]
                )
                sim_args_over = tuple(
                    t[v, val_idx][:, None] for t in validation_set_over[1:]
                )

                BPS.set_constellation(
                    model.get_constellation(
                        tuple([s[0].item() for s in sim_args])
                    ).flatten()
                )

                if args.plot:
                    mokka.utils.plot_constellation(ax1, BPS.symbols)
                    plt.show(block=False)
                    plt.pause(0.01)
                if args.estimation == "under":
                    x = model(
                        validation_set[0][val_idx, :],
                        *sim_args_under,
                    )
                elif args.estimation == "over":
                    x = model(
                        validation_set[0][val_idx, :],
                        *sim_args_over,
                    )
                else:
                    x = model(
                        validation_set[0][val_idx, :],
                        *sim_args,
                    )
                y = PN_Wiener(
                    x,
                    sim_args[0],
                    sim_args[1][0]
                )
                # Perform correction with known symbols in the stream
                if pilot:
                    phase_est = torch.angle(
                        torch.multiply(
                            x.squeeze()[::pilot_interval],
                            torch.conj(y[::pilot_interval]),
                        )
                    )
                    phase_est = torch.concat(
                        (
                            torch.zeros((5,), device=phase_est.device),
                            phase_est,
                            torch.zeros((5,), device=phase_est.device),
                        )
                    )
                    phase_est_cum = torch.cumsum(phase_est, 0)
                    phase_est = (phase_est_cum[10:] - phase_est_cum[:-10]) / 10
                    phase_est_extended = torch.repeat_interleave(
                        phase_est, pilot_interval
                    )
                    if phase_est_extended.shape[0] > y.shape[0]:
                        phase_est_extended = phase_est_extended[
                            : y.shape[0] - phase_est_extended.shape[0]
                        ]
                    y = y * torch.polar(
                        torch.full_like(phase_est_extended, 1.0), phase_est_extended
                    )

                if args.plot:
                    ax4.clear()
                    ax4.plot(
                        torch.angle(y * torch.conj(x.squeeze()))
                        .detach()
                        .cpu()
                        .numpy()
                        .flatten()[2000:2200]
                    )
                    if pilot:
                        ax4.plot(
                            phase_est_extended.detach()
                            .cpu()
                            .numpy()
                            .flatten()[2000:2200]
                        )
                    ax4.set_title("Phase offset")
                    ax4.set_ylim(-3.14 / 4, 3.14 / 4)
                    fig.canvas.draw()
                    plt.show(block=False)
                    plt.pause(0.001)

                y = BPS(y)[0]
                # During validation /eval mode LLRs are flipped
                # to agree with typical communication engineering definition
                # val_loss = loss_fn(-1 * xhat, validation_set[0][val_idx, :]).item()
                # For validation cutout first window and last window
                val_x = x.flatten()[window_size_BPS:-window_size_BPS]
                val_y = y.flatten()[window_size_BPS:-window_size_BPS]

                if args.demapper:
                    if args.demapper_type == "mismatched":
                        optim = torch.optim.Adam(demapper.parameters(), lr=1e-3)
                        for i in torch.arange(100):
                            #                            logger.info("Optimize Gaussian demapper: %s/%s", i.item()+1, 100)
                            xhat = demapper(
                                y,
                                *tuple(
                                    t[v, val_idx][:, None] for t in validation_set[1:]
                                ),
                            )
                            val_xhat = xhat[window_size_BPS:-window_size_BPS]
                            val_bmi = mokka.inft.torch.BMI(
                                m,
                                len(val_xhat.flatten()) / m,
                                validation_set[0][val_idx, :][
                                    window_size_BPS:-window_size_BPS
                                ].flatten(),
                                val_xhat.flatten(),
                            )

                            loss = m - val_bmi
                            loss.backward(retain_graph=True)
                            optim.step()
                            optim.zero_grad()
                        xhat = demapper(
                            y,
                            *tuple(t[v, val_idx][:, None] for t in validation_set[1:]),
                        )
                        val_xhat = xhat[window_size_BPS:-window_size_BPS]
                        val_bmi = mokka.inft.torch.BMI(
                            m,
                            len(val_xhat.flatten()) / m,
                            validation_set[0][val_idx, :][
                                window_size_BPS:-window_size_BPS
                            ].flatten(),
                            val_xhat.flatten(),
                        )
                    else:
                        if args.estimation == "under":
                            xhat = demapper(
                                y,
                                *tuple(
                                    t[v, val_idx][:, None]
                                    for t in validation_set_under[1:]
                                ),
                            )
                        elif args.estimation == "over":
                            xhat = demapper(
                                y,
                                *tuple(
                                    t[v, val_idx][:, None]
                                    for t in validation_set_over[1:]
                                ),
                            )
                        else:
                            xhat = demapper(
                                y,
                                *tuple(
                                    t[v, val_idx][:, None] for t in validation_set[1:]
                                ),
                            )
                        val_xhat = -1 * xhat[window_size_BPS:-window_size_BPS]
                        val_bmi = mokka.inft.torch.BMI(
                            m,
                            len(val_xhat.flatten()) / m,
                            validation_set[0][val_idx, :][
                                window_size_BPS:-window_size_BPS
                            ].flatten(),
                            val_xhat.flatten(),
                        )
                    val_bmi = val_bmi.detach().cpu().numpy()

                # ax2.clear()
                # ax2.plot(torch.abs(val_y - val_x).detach().cpu().numpy().flatten())
                # plt.show(block=False)
                # plt.pause(0.01)
                # Calculate variance of "assumed" Gaussian noise
                # First remove mean (we know it therefore no bias correction needed :) )
                val_noise = torch.mean(
                    (val_y.imag - val_x.imag) ** 2 + (val_y.real - val_x.real) ** 2
                )
                logger.debug(
                    "Validation noise after BPS: %s", np.sqrt(val_noise.item())
                )
                logger.debug("AWGN noise at channel input: %s", sim_args[0][0])
                if args.demapper:
                    val_bmi_loop.append(val_bmi)
                # val_bmis.append(val_bmi)
            if args.demapper:
                all_val_bmis.append(val_bmi_loop)
            if args.plot:
                ax2.clear()
                ax2.plot(torch.abs(val_y - val_x).detach().cpu().numpy().flatten())
                ax3.clear()
                # ax3.plot(validation_lw[:, 0].detach().cpu().numpy(), val_bmis)
                ax3.scatter(
                    y.real.detach().cpu().numpy(),
                    y.imag.detach().cpu().numpy(),
                    c=np.packbits(
                        np.reshape(
                            validation_set[0][val_idx, :]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8),
                            (-1, model.m.item()),
                        ),
                        axis=1,
                        bitorder="little",
                    ).flatten(),
                    cmap="gist_rainbow",
                    s=4,
                )

                plt.show(block=False)
                plt.pause(0.01)
            if args.output_file is not None:
                result = [
                    " ".join(["snr", "linewidth", "mean", "stddev"]) + "\n",
                ]
                if args.demapper:
                    result_values = all_val_bmis
                for run_num, run in enumerate(result_values):
                    result.append(
                        " ".join(
                            [
                                f"{10 * np.log10(1/validation_data[0][run_num][0].item()):.2f}",
                                f"{(validation_data[1][run_num][0].item() ** 2)/(2 * np.pi * T_sym):.2f}",
                                str(np.mean(run)),
                                str(np.std(run)),
                            ]
                        )
                        + "\n"
                    )
                    with open(f"./run_{run_num}.txt", "w") as run_file:
                              run_file.write("\n".join(list(str(r) for r in run)))
                with open(args.output_file, "w") as result_file:
                    result_file.writelines(result)

    return True


if __name__ == "__main__":
    sys.exit(not main())
