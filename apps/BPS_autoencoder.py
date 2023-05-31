#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import mokka
import torch
import logging
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger("mokka.BPS_autoencoder")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--model_file", help="Save file for model parameters", required=False
    )
    parser.add_argument(
        "-r",
        "--loss_file",
        help="Save loss array for evaluation of training",
        required=False,
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of epochs", type=int, default=100
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Debug mode", default=False
    )
    parser.add_argument(
        "--log-filter", default="", help="Filter specific logging modules"
    )

    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot results", default=False
    )
    parser.add_argument(
        "--demapper_type",
        default="neural",
        help="Which demapper to use",
        required=False,
    )
    parser.add_argument(
        "--plot_background",
        action="store_true",
        help="Plot results into agg backend",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--modulation",
        type=int,
        default=6,
        help="Bits/symbol of the resulting modulation",
    )
    parser.add_argument(
        "-v",
        "--verify",
        action="store_true",
        help="Use trained weights for verification",
        default=False,
    )
    parser.add_argument(
        "--no_info",
        help="Do not train with known channel info at transmitter and receiver",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--mapper-type",
        help="Which mapper to train",
        default="nn",
        choices=["nn", "simple", "separated", "separated_qam", "qam"],
    )
    parser.add_argument(
        "--demapper-depth", help="Layer depth in the demapper", type=int, default=3
    )
    parser.add_argument(
        "--demapper-width", help="Layer width in the demapper", type=int, default=128
    )
    parser.add_argument(
        "--qam-pretrained-demapper",
        help="File to weights for pre-trained QAM demapper",
        default=None,
    )
    parser.add_argument(
        "-s", "--snr", help="Which SNR to train on", default=17, type=int
    )
    parser.add_argument(
        "--max-snr",
        help="Maximum SNR to train on, if it is specified a range between \
        --snr and --max-snr is assumed",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-l", "--linewidth", help="Linewidth in kHz", default=100, type=float
    )
    parser.add_argument(
        "--max-linewidth",
        help="Maximum Linewidth, if it is specified a range between --linewidth \
        and --max-linewidth is assumed",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--channel-type",
        help="Which channel to use during training of mapper and demapper",
        default="diff_bps",
        choices=("diff_bps", "rpn"),
    )
    parser.add_argument(
        "--qam-init",
        help="Initialize Mapper with QAM",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--qam-epochs",
        help="Number of epochs to train demapper If qam-init is configured",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--pcs",
        help="Also perform probabilistic constellation shaping",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--pcs-type",
        help="Which type of PCS to perform",
        default="nn",
        choices=("nn", "mb"),
    )
    return parser.parse_args()


def pcs_loss(p_syms, m, N, bits, logits):
    return m - mokka.inft.torch.BMI(m, N, bits, logits, p_syms)


def plot_channel_output(ax, model, b, *args):
    """ """
    with torch.no_grad():
        x = model.mapper(b, *args)
        y = model.channel(x, *args)
    y_cpu = y.detach().cpu().numpy().flatten()
    ax.clear()
    ax.set_facecolor("black")
    # ax2.scatter(y_cpu.real, y_cpu.imag, c=bits, cmap="tab20", s=4)
    ax.scatter(y_cpu.real, y_cpu.imag, color="gold", alpha=0.4, s=4, edgecolor=None)


def main():  # noqa
    args = parse_args()
    if args.debug:
        mokka.utils.setup_logging(logging.DEBUG, args.log_filter)
    else:
        mokka.utils.setup_logging(logging.INFO, args.log_filter)

    model_file = args.model_file
    m = args.modulation
    # Phase noise properties
    delta_f = args.linewidth * 1000  # linewidth [Hz]
    # delta_f = 0
    sym_rate = 32e9  # symbol rate [Hz]
    T_sym = 1 / sym_rate  # symbol duration [s]
    sigma_phi = np.sqrt(2 * np.pi * delta_f * T_sym)
    max_delta_f = None
    max_sigma_phi = None
    if args.max_linewidth is not None:
        max_delta_f = args.max_linewidth * 1000
        max_sigma_phi = np.sqrt(2 * np.pi * max_delta_f * T_sym)

    # Blind Phase Search properties
    Mtestangles = 60  # Number of test angles: 36
    window_size_BPS = 128  # Number of samples used as sliding windows: 40
    no_sections = 1
    start_phase_width = 2 * np.pi
    start_phase_init = 0
    # Additive White Gaussian Noise (AWGN) properties
    SNR_dB = args.snr
    SNR_lin = 10 ** (SNR_dB / 10)
    sigma_n = np.sqrt(1 / (SNR_lin))
    max_SNR_dB = None
    max_sigma_n = None
    if args.max_snr is not None:
        max_SNR_dB = args.max_snr
        max_SNR_lin = 10 ** (max_SNR_dB / 10)
        max_sigma_n = np.sqrt(1 / (max_SNR_lin))

    pcs = args.pcs
    if pcs:
        loss_fn_name = "pcs_loss"
    else:
        loss_fn_name = "BCEWithLogitsLoss"

    # Training parameters
    training = True
    train_bps = False
    num_epochs = args.epochs
    batch_num_start = 10
    batch_num_end = 500
    batch_size_start = 1000
    batch_size_end = 10000
    learning_rate_per_epoch = 1e-3  # np.linspace(0.001, 0.0001, num=num_epochs)
    temperature_start = 1.0 / (2 * window_size_BPS)
    temperature_end = 0.001 / (2 * window_size_BPS)
    qam_pretrained_demapper = args.qam_pretrained_demapper

    config = {
        "bits_per_symbol": args.modulation,
        "demap_type": args.demapper_type,
        "demap_depth": args.demapper_depth,
        "demap_width": args.demapper_width,
        "mapper_type": args.mapper_type,
        "optimizer": "Adam",
        "loss": loss_fn_name,
        "metric": {"goal": "minimize", "name": "loss"},
        "epoch": num_epochs,
        "batch_num_start": batch_num_start,
        "batch_num_end": batch_num_end,
        "batch_size_start": batch_size_start,
        "batch_size_end": batch_size_end,
        "temperature_start": temperature_start,
        "temperature_end": temperature_end,
        "learning_rate": learning_rate_per_epoch,
        "delta_f": delta_f,
        "sigma_phi": sigma_phi,
        "SNR_dB": SNR_dB,
        "sigma_n": sigma_n,
        "no_info": args.no_info,
        "max_SNR_dB": max_SNR_dB,
        "max_sigma_n": max_sigma_n,
        "max_delta_f": max_delta_f,
        "max_sigma_phi": max_sigma_phi,
        "channel_type": args.channel_type,
        "qam_init": args.qam_init,
        "qam_epochs": args.qam_epochs,
        "qam_pretrained_demapper": args.qam_pretrained_demapper,
        "pcs": pcs,
        "pcs_type": args.pcs_type,
        "train_bps": train_bps,
    }
    if config["mapper_type"] == "qam":
        no_sections = 4
        Mtestangles = int(Mtestangles // 4)
        start_phase_width = 0
        start_phase_init = 0

    if args.plot_background:
        matplotlib.use("agg")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    batches_per_epoch = torch.linspace(
        config["batch_num_start"],
        config["batch_num_end"],
        config["epoch"],
        dtype=torch.int64,
    )
    batch_size_per_epoch = torch.linspace(
        config["batch_size_start"], config["batch_size_end"], config["epoch"]
    )
    temperature_per_epoch = torch.linspace(
        config["temperature_start"],
        config["temperature_end"],
        config["epoch"],
        device=device,
    )
    sigma_phi = config["sigma_phi"]
    sigma_n = config["sigma_n"]
    # Initialize constellation to None and then set after
    # initalizing modulator of Autoencoder

    extra_params = []
    default_extra_params = []
    if not config["no_info"]:
        if config["max_SNR_dB"] is not None:
            extra_params.append(0)
            default_extra_params.append(config["sigma_n"] ** 2)
        if config["max_sigma_phi"] is not None:
            extra_params.append(1)
            default_extra_params.append(config["sigma_phi"])
        # if config["max_lambda"] is not None:
        #     extra_params.append(2)

    if config["mapper_type"] == "nn":
        mapper = mokka.mapping.torch.ConstellationMapper(
            m, mod_extra_params=extra_params, qam_init=config["qam_init"]
        )
    elif config["mapper_type"] == "simple":
        mapper = mokka.mapping.torch.SimpleConstellationMapper(m)
    elif config["mapper_type"] == "separated":
        mapper = mokka.mapping.torch.SeparatedConstellationMapper(m)
    elif config["mapper_type"] == "separated_qam":
        mapper = mokka.mapping.torch.SeparatedConstellationMapper(m, qam_init=True)
    elif config["mapper_type"] == "qam":
        mapper = mokka.mapping.torch.QAMConstellationMapper(m)

    if config["channel_type"] == "diff_bps":

        class BPS_channel_container(torch.nn.Module):
            def __init__(self):
                super(BPS_channel_container, self).__init__()
                self.phasenoise = mokka.channels.torch.PhasenoiseWiener(
                    start_phase_width=start_phase_width,
                    start_phase_init=start_phase_init,
                )
                self.BPS = mokka.synchronizers.phase.torch.BPS(
                    Mtestangles,
                    mapper.get_constellation(*default_extra_params),
                    window_size_BPS,
                    training,
                    temperature_per_epoch[0],
                    no_sections,
                    trainable=config["train_bps"],
                    avg_filter_type="rect",
                )

            def forward(self, x, *args):
                x = self.phasenoise(x, args[0], args[1][0])
                x = self.BPS(x, *args)
                return x[0]

        channel = BPS_channel_container().to(device)
    elif config["channel_type"] == "rpn":

        class AWGN_RPN_channel_container(torch.nn.Module):
            def __init__(self):
                super(AWGN_RPN_channel_container, self).__init__()
                self.awgn = mokka.channels.torch.ComplexAWGN()
                self.rpn = mokka.channels.torch.Phasenoise()

            def forward(self, x, *args):
                # We know args[0] is N0, args[1] is RPN
                y = self.awgn(x, args[0])
                y = self.rpn(y, args[1])
                return y

        channel = AWGN_RPN_channel_container().to(device)

    model = mokka.e2e.torch.BitwiseAutoEncoder(
        m,
        channel=channel,
        mod_extra_params=extra_params,
        demod_extra_params=extra_params,
        mapper=mapper,
    )

    if config["demap_type"] == "mismatched":
        qam_constellation = torch.tensor(
            mokka.classical.QAM(config["bits_per_symbol"]).get_constellation()
        )
        demapper = mokka.mapping.torch.ClassicalDemapper(
            0.1 * config["sigma_n"] / 2,
            qam_constellation,
            model.mapper.get_constellation(default_extra_params),
            optimize=True,
        )
        model.demapper = demapper
    elif config["demap_type"] == "separated":
        if config["qam_pretrained_demapper"] is not None:
            demapper = mokka.mapping.torch.SeparatedSimpleDemapper.load_model(
                torch.load(config["qam_pretrained_demapper"], map_location=device)
            ).to(device)
        else:
            demapper = mokka.mapping.torch.SeparatedSimpleDemapper(
                m, config["demap_width"], demod_extra_params=extra_params
            )
        model.demapper = demapper
    elif config["demap_type"] == "qam":
        # demapper = mokka.mapping.torch.GaussianDemapper(
        #     torch.tensor(
        #         mokka.classical.QAM(config["bits_per_symbol"]).get_constellation(),
        #         dtype=torch.complex64,
        #     )
        # )
        demapper = mokka.mapping.torch.ClassicalDemapper(
            config["sigma_n"] / 2,
            torch.tensor(
                mokka.classical.QAM(config["bits_per_symbol"]).get_constellation(),
                dtype=torch.complex64,
            )
            .squeeze()
            .to(device),
            optimize=True,
        )
        model.demapper = demapper
    else:
        if config["qam_pretrained_demapper"] is not None:
            demapper = mokka.mapping.torch.ConstellationDemapper.load_model(
                torch.load(config["qam_pretrained_demapper"], map_location=device)
            ).to(device)
        else:
            demapper = mokka.mapping.torch.ConstellationDemapper(
                config["bits_per_symbol"],
                config["demap_depth"],
                config["demap_width"],
                demod_extra_params=extra_params,
            )
        model.demapper = demapper
    optim_parameters = []
    if pcs:
        #        if config["mapper_type"] == "qam":
        if config["pcs_type"] == "mb":
            pcs_sampler = mokka.mapping.torch.MBPCSSampler(
                model.mapper.get_constellation(*default_extra_params),
                pcs_extra_params=extra_params,
                fixed_lambda=False,
            ).to(device)
        elif config["pcs_type"] == "nn":
            pcs_sampler = mokka.mapping.torch.PCSSampler(
                config["bits_per_symbol"], pcs_extra_params=extra_params
            ).to(device)
        else:
            raise ValueError("pcs-type wrong")
        optim_parameters.append({"params": pcs_sampler.parameters()})
    model = model.to(device)
    if config["channel_type"] == "diff_bps":
        channel.BPS.set_constellation(
            model.mapper.get_constellation(*default_extra_params).flatten()
        )
    if config["train_bps"]:
        optim_parameters.append({"params": channel.BPS.parameters()})
    if not args.verify:
        model.train()
        # Define training
        if not config["qam_init"]:
            optim_parameters.append({"params": model.mapper.parameters()})
        optim_parameters.append({"params": model.demapper.parameters()})
        optim = getattr(torch.optim, config["optimizer"])(
            optim_parameters, lr=learning_rate_per_epoch
        )
        max_size = int(torch.max(batch_size_per_epoch) * torch.max(batches_per_epoch))
        training_bits = torch.tensor(
            mokka.utils.generators.numpy.generate_bits(
                (max_size, config["bits_per_symbol"])
            ),
            dtype=torch.float32,
        ).to(device)
        # Only used for fixed SNR & linewidth
        training_N0 = torch.full((max_size,), config["sigma_n"] ** 2, device=device)[
            :, None
        ]
        training_lw = torch.full((max_size,), config["sigma_phi"], device=device)[
            :, None
        ]
        training_idx = np.arange(training_bits.shape[0], dtype=int)
        total_loss = []
        min_loss = np.infty
        if pcs:
            loss_fn = pcs_loss
        else:
            loss_fn = getattr(torch.nn, config["loss"])()
        optim.zero_grad()
        if config["max_SNR_dB"] is not None:
            batch_N0_gen = torch.distributions.Uniform(
                config["max_sigma_n"] ** 2, config["sigma_n"] ** 2
            )

        if config["max_sigma_phi"] is not None:
            # Get linewidth per batch
            batch_sigma_phi_gen = torch.distributions.Uniform(
                config["sigma_phi"], config["max_sigma_phi"]
            )

        for e in range(config["epoch"]):
            if e == config["qam_epochs"] and config["qam_init"]:
                optim = getattr(torch.optim, config["optimizer"])(
                    model.parameters(), lr=learning_rate_per_epoch
                )
                channel.BPS.Mtestangles = Mtestangles * 4
                channel.phasenoise.start_phase_width = start_phase_width = 2 * np.pi
                channel.BPS.no_sectors = 1

            if config["channel_type"] == "diff_bps" and not config["train_bps"]:
                channel.BPS.temperature_per_epoch = temperature_per_epoch[e]
            if args.plot:
                if config["channel_type"] == "diff_bps":
                    mokka.utils.plot_constellation(
                        ax1,
                        channel.BPS.symbols,
                        size=(model.mapper.p_symbols * 600).detach().cpu().numpy(),
                    )
                else:
                    mokka.utils.plot_constellation(
                        ax1,
                        model.mapper.get_constellation(
                            training_N0[0].item(), training_lw[0].item()
                        ),
                    )
                fig.suptitle(
                    f"SNR: {-10 * np.log10(training_N0[0].item()):.2f} dB - Linewidth:"
                    f" {(training_lw[0].item() ** 2)/(2 * np.pi * T_sym * 1000):.2f} kHz"
                )
            np.random.shuffle(training_idx)
            logger.info("Epoch %s/%s", e + 1, config["epoch"])
            epoch_loss = []
            num_batches = batches_per_epoch[e]
            for batch_no in range(num_batches):
                batch_idx = training_idx[
                    batch_no
                    * int(batch_size_per_epoch[e]) : min(
                        (batch_no + 1) * int(batch_size_per_epoch[e]),
                        training_idx.shape[0],
                    )
                ]

                if config["demap_type"] == "mismatched":
                    with torch.no_grad():
                        model.demapper.update_constellation(
                            model.mapper.get_constellation(default_extra_params)
                        )
                if config["max_SNR_dB"] is not None:
                    training_N0 = torch.full(
                        (max_size,), batch_N0_gen.sample(), device=device
                    )[:, None]
                if config["max_sigma_phi"] is not None:
                    training_lw = torch.full(
                        (max_size,), batch_sigma_phi_gen.sample(), device=device
                    )[:, None]
                # if config["max_lambda"] is not None:
                #     training_lambda = torch.full(
                #         (max_size,), batch_lambda_gen.sample(), device=device
                #     )[:, None]

                if config["pcs"]:
                    # mapper.p_symbols = pcs_sampler.p_symbols(
                    #     training_N0[batch_idx, 0], training_lw[batch_idx, 0]
                    # ).detach()
                    # QAM does scale the constellation, but that's super annoying for finding the correct lambda
                    # pcs_sampler.symbols = model.mapper.get_constellation(training_N0[0,0], training_lw[0,0]).flatten()
                    # if config["max_lambda"] is not None:
                    #     pcs_sampler._lambda = training_lambda[0][0]
                    mapper.p_symbols = pcs_sampler.p_symbols(
                        training_N0[batch_idx, 0], training_lw[batch_idx, 0]
                    )
                    batch_set = (
                        mokka.utils.bitops.torch.idx2bits(
                            pcs_sampler(
                                int(batch_size_per_epoch[e]),
                                training_N0[batch_idx, :],
                                training_lw[batch_idx, :],
                            )[
                                torch.randperm(batch_size_per_epoch[e].long().item()),
                            ],
                            config["bits_per_symbol"],
                        ).float(),
                        training_N0[batch_idx, :],
                        training_lw[batch_idx, :],
                        # training_lambda[batch_idx, :],
                    )
                else:
                    batch_set = (
                        training_bits[batch_idx, :],
                        training_N0[batch_idx, :],
                        training_lw[batch_idx, :],
                    )
                if config["channel_type"] == "diff_bps":
                    if not config["no_info"]:
                        channel.BPS.set_constellation(
                            model.mapper.get_constellation(
                                training_N0[0], training_lw[0]
                            ).flatten()
                        )
                    else:
                        channel.BPS.set_constellation(
                            model.mapper.get_constellation().flatten()
                        )
                if args.plot:
                    if batch_no == 0:
                        with torch.no_grad():
                            plot_channel_output(
                                ax2,
                                model,
                                batch_set[0],
                                *tuple(t for t in batch_set[1:]),
                            )
                            if not args.plot_background:
                                plt.show(block=False)
                                plt.pause(0.01)
                yhat = model.forward(
                    batch_set[0],
                    *tuple(t for t in batch_set[1:]),
                )
                yhat_no_resid = yhat[window_size_BPS:-window_size_BPS, :]
                bmi = mokka.inft.torch.BMI(
                    config["bits_per_symbol"],
                    len(yhat_no_resid.flatten()) / config["bits_per_symbol"],
                    batch_set[0][window_size_BPS:-window_size_BPS].flatten().detach(),
                    -1 * yhat_no_resid.flatten(),
                    p=model.mapper.p_symbols.detach()
                    if config["mapper_type"] == "nn"
                    else torch.tensor(
                        2 ** config["bits_per_symbol"]
                        * [
                            1.0 / (2 ** config["bits_per_symbol"]),
                        ]
                    ),
                )
                if torch.allclose(bmi, torch.tensor(0.0)):
                    print(yhat)
                if pcs:
                    loss = loss_fn(
                        model.mapper.p_symbols,
                        config["bits_per_symbol"],
                        len(yhat_no_resid.flatten()) / config["bits_per_symbol"],
                        batch_set[0][window_size_BPS:-window_size_BPS],
                        -1 * yhat_no_resid,
                    )
                else:
                    loss = loss_fn(
                        yhat_no_resid,
                        batch_set[0][window_size_BPS:-window_size_BPS],
                    )
                loss.backward()
                optim.step()
                optim.zero_grad()
                epoch_loss.append(float(loss.detach().cpu().numpy()))

                if config["pcs"]:
                    logger.debug(
                        "Probabilities: %s",
                        model.mapper.p_symbols,
                    )

                if not torch.any(torch.isnan(loss)):
                    if (
                        model_file is not None
                        and loss.detach().cpu().numpy() < min_loss
                    ):
                        torch.save(model.state_dict(), model_file)
                        if config["pcs"]:
                            torch.save(
                                pcs_sampler.state_dict(),
                                model_file.split(".")[0] + "_pcs.bin",
                            )
                        if config["train_bps"]:
                            torch.save(
                                channel.BPS.state_dict(),
                                model_file.split(".")[0] + "_bps.bin",
                            )
                        min_loss = loss.detach().cpu().numpy()
                    if args.loss_file is not None:
                        np.save(args.loss_file, np.array(total_loss))
                else:
                    logger.warning("Gradients exploded, loss contains nan")
                    return False
            logger.info("epoch avg loss: %s", np.mean(epoch_loss))
            logger.info("last batch bmi: %s", bmi.item())
            # scheduler.step()
            # logger.debug("LR: %s", scheduler.get_last_lr())
            total_loss.append(epoch_loss)
        if args.loss_file is not None:
            np.save(args.loss_file, np.array(total_loss))
    if args.plot and not args.plot_background:
        plt.show()
    return True


if __name__ == "__main__":
    sys.exit(not main())
