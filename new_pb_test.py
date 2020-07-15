import torchvision.datasets as ds
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision
from torch.backends import cudnn
cudnn.benchmark = True
import inspect
from torchsummary import summary
import utils.helperFunctions as hf

def get_bad_kws(func, **kwargs):
    """ Parse kws in kwargs against the arguments of func and return a list of
        incompatible kws
    """
    return [key for key in kwargs.keys() if key not in inspect.getfullargspec(func).args]


def select_data_set(ds_name, *args, **kwargs):
    """Select a dataset based off of name and pass on arguments"""
    if not hasattr(ds, ds_name):
        raise TypeError('Bad key: %s, not in the dict' % ds_name)
    return getattr(ds, ds_name)(*args, **kwargs)
    # dataset_func = getattr(ds, ds_name)
    # if len(get_bad_kws(dataset_func, **kwargs)) > 0:
    #     bad_args = get_bad_kws(dataset_func, **kwargs)
    #     raise TypeError('\n\t\tThe keyword(s) "%s" is/are not valid for %s' % ('", "'.join(bad_args), str(dataset_func))
    #                      + '\n\t\tValid Keys are: %s' % (', '.join(inspect.getfullargspec(dataset_func).args)))
    # return dataset_func(*args, **kwargs)


def select_from_module(module, attr_, *args, **kwargs):
    if not hasattr(module, attr_):
        raise TypeError('Bad attr. key: %s, not in the %s' % (attr_, str(module)))
    return getattr(module, attr_)(*args, **kwargs)


def select_module(module, attr_):
    """ Select a function (attr_) from a module obj (module)"""
    if not hasattr(module, attr_):
        raise TypeError('Bad attr. key: %s, not in the %s' % (attr_, str(module)))
    return getattr(module, attr_)


def apply_arguments(func, *args, **kwargs):
    """ Apply arguments / kw arguments to a function object """
    return func(*args, **kwargs)


def select_torch_model(tag, *args, **kwargs):
    if tag not in torchvision.models.__dict__ or tag[:2] == '__' or not tag.islower():
        s = 'invalid name network name given: %s\n' % (tag) +'Lower: %s\n' % (tag.islower()) + \
            'In __dict__: %s\n' % (tag in torchvision.models.__dict__) \
            + 'Doesnt begin with __: %s ' % str(not(tag[:2] == '__'))
        raise TypeError(s)
    return torchvision.models.__dict__[tag](*args, **kwargs)

@hf.calculate_time
def main():
    import torch.nn as nn

    init_tag = 'xavier_normal_'


    playbook = hf.jsonLoad('playbooks/PlaybookTemplate.json')
    transform_list = playbook['dataset'][2]['transform']
    compose_list = []
    for key, args, kwargs in transform_list:
        compose_list.append(select_module(transforms, key)(*args, **kwargs))
    print(transforms.Compose(compose_list))


    # dataset_tag = playbook['dataset'][0]
    tag_ds, args_ds, kws_ds = playbook['dataset']
    kws_ds['transform'] = transforms.Compose(compose_list)
    tag_model, args_model, kws_model = playbook['model']
    tag_sched, args_sched, kws_sched = playbook['scheduler']
    tag_init, args_init, kws_init = playbook['init']
    tag_optim, args_optim, kws_optim = playbook['optim']
    tad_loader, args_loader, kws_loader = playbook['loader']


    # 1. Select Model
    base_model = select_module(torchvision.models, tag_model)(*args_model, **kws_model)
    model = nn.Sequential(base_model, nn.Linear(1000, 10))
    base_model.cuda()
    model.cuda()
    # 2. Select init func & initialize model params
    init_func = select_module(nn.init, tag_init)
    for param in model.parameters():
        if len(param.shape) > 1:
            init_func(param, *args_init, **kws_init)
    # 3. Select optimizer
    optim_obj = select_module(torch.optim, tag_optim)(model.parameters(), *args_optim, **kws_optim)
    # 4. Select Dataset, Scheduler, and Loader
    dataset_obj = select_module(torchvision.datasets, tag_ds)(**kws_ds)
    scheduler = select_module(torch.optim.lr_scheduler, tag_sched)(optim_obj, *args_sched,** kws_sched)
    loader = data.DataLoader(dataset_obj, *args_loader, **kws_loader)

    loss_function = torch.nn.CrossEntropyLoss()
    model.cuda()


    # Training Loop
    total_predicted = 0
    num_correct = 0

    input, labels = None, None
    for epoch in list(range(10)):
        epoch_loss = 0
        num_correct = 0
        total_predicted = 0
        for i, batch_data in enumerate(loader, 0):

            inputs, labels = batch_data[0].cuda(), batch_data[1].cuda()
            optim_obj.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optim_obj.step()

            # pred_labels = [torch.argmax(output).item() for output in outputs]
            pred_labels = torch.argmax(outputs, axis=1)
            num_correct += len([(a, b) for a, b in zip(labels, pred_labels) if a == b])
            total_predicted += 64
            epoch_loss += loss

            if i % 50 == 5:
                print('Batch %s, Accuracy: %4.3f' % (str(i), num_correct/total_predicted))

        print("Epoch Loss: %4.4f" % epoch_loss)
        scheduler.step()

if __name__ == "__main__":
    main()
x = {'transforms': [{'ToTensor': {'args': [], 'kwargs': {}}}]}
exit()

'transforms.Compose([transforms.ToTensor()])'




# transforms.Compose[select_module(torchvision.transforms, key)(*args, **kwargs) for key in transform_keys]
# for attr_, val in torchvision.models.__dict__.items():
#     print(hasattr(getattr(torchvision.models, attr_), 'num_classes'), attr_)

#
# print(len(dataset_obj.classes))
# print('Dataset Obj: ', dataset_obj)
# print('     Model Obj: ', str(model.state_dict().keys()))
# # summary(model, input_size=(3, 32, 32))
# print('    Loader Obj: ', loader)
# print('      Init Obj: ', init_func)
# print(' Scheduler Obj: ', scheduler)
# print('    Loader Obj: ', loader)