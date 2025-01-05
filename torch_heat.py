import numpy as np
import torch
from torch import nn
from torch.autograd import grad
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def f(x: torch.tensor, y: torch.tensor):
    return 3*x**2

def g(x: torch.tensor, y: torch.tensor):
    return - y 

def h(x: torch.tensor):
    return torch.sin(8 * np.pi * x)

def zero(x: torch.tensor):
    return 0 * x

class SimpleNN(nn.Module):
    def __init__(self, tdim: int, xdim: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(tdim + xdim, hidden),
            nn.Tanh(),     
            nn.Linear(hidden, 30),
            nn.Tanh(),
            nn.Linear(30, 1),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

    def show_parameters(self):
        for paramter in self.parameters():
            print(paramter)
    
class ODE_solver_o1(SimpleNN):
    def __init__(self, hidden: int, f: callable, batch: int, learning_rate: float, weight_decay: float, momentum: float, ic: float):
        super().__init__(0,1, hidden)
        self.f = f
        self.batch = batch
        #self.optimizer = torch.optim.SGD(self.parameters(), learning_rate, weight_decay, momentum)
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate, amsgrad=True, weight_decay=0.00005)
        self.ic = ic

    def loss(self, x: torch.tensor, y: torch.tensor):
        dydx = grad(y, x, grad_outputs=y.data.new(y.shape).fill_(1), create_graph=True, retain_graph=True)
        error = dydx[0] - self.f(x, y)
        ic_error = self(torch.zeros(1)) - self.ic
        return torch.mean(error ** 2) + ic_error ** 2
    
    def loss2(self, x: torch.tensor, y: torch.tensor):
        dydx = grad(y, x, grad_outputs=y.data.new(y.shape).fill_(1), create_graph=True, retain_graph=True)
        partial2x = grad(dydx[0], x, grad_outputs=y.data.new(y.shape).fill_(1), create_graph=True, retain_graph=True)
        error = partial2x[0] - self.f(x, y)
        zero =  torch.zeros((1,1), requires_grad=True)
        ic_pred = self(zero)
        ic_error = ic_pred - 0
        ic1_pred = grad(ic_pred, zero, create_graph=True, retain_graph=True)
        ic1_error = ic1_pred[0] - 1
        return torch.mean(error ** 2) + ic_error ** 2 + ic1_error ** 2
    
    def training_loop(self, order: int):
        self.train()
        X = 2 * np.pi * torch.rand(self.batch, 1, requires_grad = True)
        pred = self(X)
        if order == 1:
            loss = self.loss(X, pred)
        elif order == 2:
            loss = self.loss2(X, pred)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def training_epochs(self, epochs: int, order: int):
        for t in range(epochs):
            loss = self.training_loop(order)
            if t % 100 == 0:
                print(f"Epoch {t+1}\n-------------------------------")
                print(f'loss: {loss.item()}')

    def graph_prediction(self):
        with torch.no_grad():
            x_values = torch.linspace(0,2 * np.pi,10000).reshape(10000,1)
            y_values = self(x_values).reshape(10000)
            #exact = np.sin(x_values.numpy())
            #exact = x_values.numpy() ** 3 / 6
            plt.plot(x_values.numpy(), y_values.numpy(), label = 'prediction')
            #plt.plot(x_values.numpy(), exact, label = 'exact')
            plt.legend()
            plt.show()

class EvolutionEquation1D(SimpleNN):
    def __init__(self, hidden: int, f: callable, batch: int, learning_rate: float, weight_decay: float, momentum: float, bc0 : float, bc1 : float, ic2 : callable = None):
        super().__init__(1, 1, hidden)
        self.f = f #initial confition u(0,x) = f(x)
        self.batch = batch
        self.batch2 = batch ** 2
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate, amsgrad=True, weight_decay=0.00005) #weight_decay, momentum)
        self.bc0 = bc0
        self.bc1 = bc1
        self.ic2 = ic2 #inital condition u_t(0,x) = ic2(x) for wave equation

    def heat_operator(self, x: torch.tensor, y: torch.tensor): #x = (t,x), y = model(x)
        gradient = grad(y, x, grad_outputs=y.data.new(y.shape).fill_(1), create_graph=True, retain_graph=True)
        partialt = torch.index_select(gradient[0], 1, torch.tensor([0]))
        partialx = torch.index_select(gradient[0], 1, torch.tensor([1]))
        partialx_grad = grad(partialx, x, grad_outputs=y.data.new(y.shape).fill_(1), create_graph=True, retain_graph=True)
        partial2x = torch.index_select(partialx_grad[0], 1, torch.tensor([1]))
        return torch.sub(partialt, partial2x)
    
    def wave_operator(self, x: torch.tensor, y: torch.tensor):
        gradient = grad(y, x, grad_outputs=y.data.new(y.shape).fill_(1), create_graph=True, retain_graph=True)
        partialt = torch.index_select(gradient[0], 1, torch.tensor([0]))
        partialt_grad = grad(partialt, x, grad_outputs=y.data.new(y.shape).fill_(1), create_graph=True, retain_graph=True)
        partial2t = torch.index_select(partialt_grad[0], 1, torch.tensor([0]))
        partialx = torch.index_select(gradient[0], 1, torch.tensor([1]))
        partialx_grad = grad(partialx, x, grad_outputs=y.data.new(y.shape).fill_(1), create_graph=True, retain_graph=True)
        partial2x = torch.index_select(partialx_grad[0], 1, torch.tensor([1]))
        return torch.sub(partial2t, partial2x)

    #x is the input coordinate (t,x0)
    #y is model(x)
    #ic_testx is the inital condition input (i.e. (0,x0) value)
    #ic_testy is model(ic_testx)
    #t0_test is the left boundary condition input (t, 0), t1_test right boundary condition (t, 1)
    #hw chooses heat equation for 'h' and wave equation for 'w'
    def loss(self, x: torch.tensor, y: torch.tensor, ic_testx: torch.tensor, ic_testy: torch.tensor, t0_test: torch.tensor, t1_test: torch.tensor,
     t0_y: torch.tensor, t1_y: torch.tensor, hw: str):
        if hw == 'heat':
            error = self.heat_operator(x,y)
        elif hw == 'wave':
            error = self.wave_operator(x,y)
        bc0_error = t0_y - self.bc0
        bc1_error = t1_y - self.bc1
        ic_error = torch.index_select(self.f(ic_testx), 1, torch.tensor([1])) - ic_testy
        if hw == 'heat':
            return   0.01 * torch.mean(error ** 2) + torch.mean(ic_error ** 2) + 1 * torch.mean(bc0_error ** 2) + 1 * torch.mean(bc1_error ** 2) 
        elif hw == 'wave':
            ic_testy_grad = grad(ic_testy, ic_testx, grad_outputs=ic_testy.data.new(ic_testy.shape).fill_(1), create_graph= True, retain_graph= True)
            partial_t_ic_test_y = torch.index_select(ic_testy_grad[0], 1, torch.tensor([0]))
            ic2_error = partial_t_ic_test_y - torch.index_select(self.ic2(ic_testx), 1, torch.tensor([1]))
            return 0.01 * torch.mean(error ** 2) + torch.mean(ic_error ** 2) + 1* torch.mean(bc0_error ** 2) + 1* torch.mean(bc1_error ** 2) + torch.mean(ic2_error ** 2)

    def training_loop(self, hw: str):
        self.train()
        X = torch.rand(self.batch2, 2, requires_grad = True)
        Y_pred = self(X)
        x0 = torch.cat((torch.zeros(self.batch,1), torch.rand(self.batch, 1, requires_grad= True)), 1)
        y0 = self(x0)
        t0 = torch.cat((torch.rand(self.batch, 1, requires_grad= True), torch.zeros(self.batch,1)) , 1)
        t0_y = self(t0)
        t1 = torch.cat((torch.rand(self.batch, 1, requires_grad= True), torch.ones(self.batch,1)) , 1)
        t1_y = self(t1)
        loss = self.loss(X, Y_pred, x0, y0, t0, t1, t0_y, t1_y, hw)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def training_epochs(self, epochs: int, hw: str):
        for t in range(epochs):
            loss = self.training_loop(hw)
            if t % 100 == 0:
                print(f"Epoch {t+1}\n-------------------------------")
                print(f'loss: {loss.item()}')

    def graph_prediction(self):
        plt.ion()
        plt.ylim(-1, 1)
        plt.xlim(0, 1)
        with torch.no_grad():
            x_values = torch.linspace(0,1,1000).reshape(1000,1)
            y_values = []
            y_exacts = []
            for i in range(1000):
                print(i)
                t = i / 1000
                x2 = torch.cat((torch.full((1000,1), t), x_values), 1)
                y_value = self(x2).reshape(1000)
                y_values.append(y_value.numpy())
                y_exact = np.cos(8 * np.pi * t) * torch.sin(8 * np.pi * x_values)
                y_exacts.append(y_exact.numpy())
            graph, = plt.plot(x_values.numpy(), y_values[0], label = 'prediction')
            exact, = plt.plot(x_values.numpy(), y_exacts[0], label = 'exact')
            for i in range(1000):
                print(i)
                plt.pause(0.001)
                graph.set_ydata(y_values[i])
                exact.set_ydata(y_exacts[i])
                plt.draw()

class NonEvolution2D(SimpleNN):
    pass


def test2():
    np.random.seed(2)
    torch.manual_seed(2)
    model = ODE_solver_o1(10, g, 10, 0.1, 0.00005, 0, ic = 1)
    epochs = 20000
    model.training_epochs(epochs)
    model.graph_prediction()

def test3():
    np.random.seed(2)
    torch.manual_seed(2)
    model = ODE_solver_o1(100, g, 10, 0.01, 0.00005, 0, ic = 1)
    epochs = 10000
    model.training_epochs(epochs, 2)
    model.graph_prediction()

def test4():
    np.random.seed(2)
    torch.manual_seed(2)
    model = EvolutionEquation1D(100, h, 10, 0.005, 0.00005, 0, bc0 = 0, bc1 = 0)
    epochs = 20000
    model.training_epochs(epochs, 'heat')
    model.graph_prediction()

def test5():
    np.random.seed(2)
    torch.manual_seed(2)
    model = EvolutionEquation1D(100, h, 10, 0.005, 0.00005, 0, bc0 = 0, bc1 = 0, ic2= zero)
    epochs = 20000
    model.training_epochs(epochs, 'wave')
    model.graph_prediction()


if __name__ == "__main__":
    test5()

    
    