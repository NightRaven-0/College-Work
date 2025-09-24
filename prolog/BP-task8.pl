% bp_simple.pl  -- small backprop demo without fancy SWI-specific features
% Works in basic Prolog interpreters (no lambda, no set_random).

% ---------- basic vector/matrix helpers ----------

% dot_product(ListA, ListB, Sum).
dot_product([], [], 0).
dot_product([A|As], [B|Bs], Sum) :-
    dot_product(As, Bs, Rest),
    Sum is Rest + A*B.

% vec_add(A,B,R)
vec_add([], [], []).
vec_add([A|As],[B|Bs],[R|Rs]) :-
    R is A + B,
    vec_add(As,Bs,Rs).

% scalar_mul(S, V, R)
scalar_mul(_, [], []).
scalar_mul(S, [X|Xs], [Y|Ys]) :-
    Y is S*X,
    scalar_mul(S, Xs, Ys).

% outer_product(A,B,Matrix)  A is length N, B length M -> Matrix N rows each of length M
outer_product([], _, []).
outer_product([A|As], B, [Row|Rows]) :-
    build_row(A, B, Row),
    outer_product(As, B, Rows).

build_row(_, [], []).
build_row(A, [B|Bs], [V|Vs]) :-
    V is A * B,
    build_row(A, Bs, Vs).

% matrix_add(A,B,C)
matrix_add([], [], []).
matrix_add([R1|Rs1], [R2|Rs2], [Rr|Rrs]) :-
    vec_add(R1, R2, Rr),
    matrix_add(Rs1, Rs2, Rrs).

% matrix_scalar_mul(S, M, R)
matrix_scalar_mul(_, [], []).
matrix_scalar_mul(S, [Row|Rows], [RowR|RowsR]) :-
    scalar_mul(S, Row, RowR),
    matrix_scalar_mul(S, Rows, RowsR).

% transpose(M, MT)
transpose([], []).
transpose([[]|_], []) :- !.
transpose(M, [Col|Cols]) :-
    first_column(M, Col, RestRows),
    transpose(RestRows, Cols).

first_column([], [], []).
first_column([[H|T]|Rows], [H|Hs], [T|Ts]) :-
    first_column(Rows, Hs, Ts).

% ---------- activations ----------

sigmoid(X, Y) :- Y is 1.0/(1.0 + exp(-X)).
sigmoid_list([], []).
sigmoid_list([X|Xs], [Y|Ys]) :-
    sigmoid(X, Y),
    sigmoid_list(Xs, Ys).

sigmoid_deriv_from_output(O, D) :- D is O * (1.0 - O).
sigmoid_deriv_list([], []).
sigmoid_deriv_list([O|Os], [D|Ds]) :-
    sigmoid_deriv_from_output(O, D),
    sigmoid_deriv_list(Os, Ds).

% ---------- network representation ----------
% net(InN, HN, OnN, Wih, Who, Bh, Bo, LR)

% ---------- deterministic init (no random) ----------
init_net(InN, HN, ON, LR, _SeedIgnored, net(InN,HN,ON,Wih,Who,Bh,Bo,LR)) :-
    % for portability we use small fixed weights based on sizes
    make_vector(HN, 0.2, Bh),
    make_vector(ON, 0.1, Bo),
    make_matrix(HN, InN, 0.15, Wih),
    make_matrix(ON, HN, -0.12, Who).

make_vector(0, _, []).
make_vector(N, Val, [Val|Vs]) :- N>0, N1 is N-1, make_vector(N1, Val, Vs).

make_matrix(0, _, _, []).
make_matrix(R, C, Val, [Row|Rows]) :-
    R>0,
    make_vector(C, Val, Row),
    R1 is R-1,
    make_matrix(R1, C, Val, Rows).

% ---------- forward pass ----------
% forward(+Net, +X, -HiddenOut, -OutputOut, -State)
forward(net(_InN,_HN,_ON,Wih,Who,Bh,Bo,_LR), X, Hout, Yout, state(X, Zin, Hout, Yin, Yout)) :-
    % Zin = Wih * X + Bh
    mult_matrix_vector(Wih, X, ZinTemp),
    vec_add(ZinTemp, Bh, Zin),
    sigmoid_list(Zin, Hout),
    % Yin = Who * Hout + Bo
    mult_matrix_vector(Who, Hout, YinTemp),
    vec_add(YinTemp, Bo, Yin),
    sigmoid_list(Yin, Yout).

% mult_matrix_vector(Matrix, Vec, Result)
mult_matrix_vector([], _, []).
mult_matrix_vector([Row|Rows], V, [S|Ss]) :-
    dot_product(Row, V, S),
    mult_matrix_vector(Rows, V, Ss).

% ---------- backprop single example ----------
% train_one(+Net, +X, +T, -NetNew)
train_one(net(InN,HN,ON,Wih,Who,Bh,Bo,LR), X, T, net(InN,HN,ON,Wih2,Who2,Bh2,Bo2,LR)) :-
    forward(net(InN,HN,ON,Wih,Who,Bh,Bo,LR), X, Hout, Yout, _State),
    % output delta: delta_o = (T - Y) * f'(Y)
    maplist_output_delta(T, Yout, DeltaO),
    % hidden delta: DeltaH_j = f'(Hj) * sum_k Who_kj * DeltaO_k
    transpose(Who, WhoT),
    mult_matrix_vector(WhoT, DeltaO, HiddenSums),
    sigmoid_deriv_list(Hout, Hder),
    elementwise_mul(Hder, HiddenSums, DeltaH),

    % weight updates
    outer_product(DeltaO, Hout, DWho),
    matrix_scalar_mul(LR, DWho, DWhoScaled),
    matrix_add(Who, DWhoScaled, Who2),

    outer_product(DeltaH, X, DWih),
    matrix_scalar_mul(LR, DWih, DWihScaled),
    matrix_add(Wih, DWihScaled, Wih2),

    % bias updates
    add_scaled_to_vector(Bo, DeltaO, LR, Bo2),
    add_scaled_to_vector(Bh, DeltaH, LR, Bh2).

maplist_output_delta([], [], []).
maplist_output_delta([T|Ts], [Y|Ys], [D|Ds]) :-
    Err is T - Y,
    sigmoid_deriv_from_output(Y, Der),
    D is Err * Der,
    maplist_output_delta(Ts, Ys, Ds).

elementwise_mul([], [], []).
elementwise_mul([A|As], [B|Bs], [C|Cs]) :-
    C is A * B,
    elementwise_mul(As, Bs, Cs).

add_scaled_to_vector(Vec, Deltas, LR, VecNew) :-
    scale_vector(Deltas, LR, Scaled),
    vec_add(Vec, Scaled, VecNew).

scale_vector([], _, []).
scale_vector([X|Xs], S, [Y|Ys]) :-
    Y is X * S,
    scale_vector(Xs, S, Ys).

% ---------- train epochs ----------
train_epochs(Net, _Data, 0, Net) :- !.
train_epochs(Net0, Data, E, NetOut) :-
    E>0,
    train_epoch(Net0, Data, Net1),
    E1 is E-1,
    train_epochs(Net1, Data, E1, NetOut).

train_epoch(Net, [], Net).
train_epoch(Net0, [(X,T)|Rest], NetOut) :-
    train_one(Net0, X, T, Net1),
    train_epoch(Net1, Rest, NetOut).

% ---------- testing helper ----------
test_net(Net, Data) :-
    forall(member((X,T), Data),
            ( forward(Net, X, _H, Yout, _S),
              write('Input: '), write(X),
              write('  Target: '), write(T),
              write('  Output: '), write(Yout), nl
            )).

% ---------- example XOR run ----------
example_xor :-
    Data = [ ([0.0,0.0],[0.0]),
             ([0.0,1.0],[1.0]),
             ([1.0,0.0],[1.0]),
             ([1.0,1.0],[0.0]) ],
    init_net(2,2,1,0.5,42,Net0),
    train_epochs(Net0, Data, 2000, NetTrained),
    write('Training finished. Testing:'), nl,
    test_net(NetTrained, Data).