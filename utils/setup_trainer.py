import chainer
from chainer import training
from chainer.training import extensions

from training.extensions.print_stack_report import PrintStackReport
from utils.slack import SlackOut


def postprocess_loss(fig, axes, obj):
    "Modify graph of loss."
    axes.set_xlim(xmin=0)
    axes.set_yscale('log')
    axes.grid(which='both')


def postprocess_accuracy(fig, axes, obj):
    "Modify graph of accuracy."
    axes.set_xlim(xmin=0)
    axes.set_ylim(ymax=1)
    axes.grid(which='both')


def setup_trainer(args, train_iter, test_iter, model):
    # Set optimizer
    if args.opt == 'NesterovAG':
        optimizer = chainer.optimizers.NesterovAG(args.lr)
        lr = 'lr'
    elif args.opt == 'Adam':
        optimizer = chainer.optimizers.Adam(
            alpha=args.lr, amsgrad=args.amsgrad, adabound=args.adabound)
        lr = 'alpha'
    optimizer.setup(model)
    if args.weight_decay != 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    # Setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.outdir)
    # Set extension: Evaluater
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),
                   name='val')
    # Set extension: learning rate decay
    points = [args.epoch // 2, args.epoch * 3 // 4]
    trigger = training.triggers.ManualScheduleTrigger(points, 'epoch')
    trainer.extend(extensions.ExponentialShift(lr, 0.1), trigger=trigger)
    # Set extension: Dump graph
    trainer.extend(extensions.dump_graph('main/loss'))
    # Set extension: Snapshot
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    # Set extension: log
    trainer.extend(extensions.LogReport())
    # Set extension: observe_lr
    trainer.extend(extensions.observe_lr())
    # Set extension: Print Report
    trainer.extend(PrintStackReport(
        ['epoch', 'lr', 'main/loss', 'val/main/loss',
         'main/accuracy', 'val/main/accuracy', 'elapsed_time']))
    if args.slack_interval:
        # Set extension: Print Report
        trainer.extend(PrintStackReport(
            ['epoch', 'lr', 'main/loss', 'val/main/loss',
             'main/accuracy', 'val/main/accuracy', 'elapsed_time'],
            out=SlackOut()), trigger=(args.slack_interval, 'epoch'))
    # Set extension: Progress bar
    trainer.extend(extensions.ProgressBar(update_interval=args.interval))
    # Set extension: Save train curve graph.
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key='epoch',
        postprocess=postprocess_loss, file_name='loss.png',
        marker='', grid=False))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key='epoch',
        postprocess=postprocess_accuracy, file_name='accuracy.png',
        marker='', grid=False))
    return trainer
