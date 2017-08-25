\LoadClass[12pt]{article}
\RequirePackage[letterpaper, left=1in, right=1in, top=1in, bottom=1.4in]{geometry}

%\RequirePackage[osf]{mathpazo}

\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{graphicx}
\RequirePackage{latexsym}
\RequirePackage[colorlinks, urlcolor=blue]{hyperref}

\setlength{\parskip}{6pt}

\usepackage{indentfirst}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\eop}{$\diamond$}
\newcommand{\var}{\textup{var}}

\newcommand{\mytitle}{CS 4444 -- Report for Homework\ \hnumber}

\newcommand{\pledge}{
\centerline{\textit{On my honor as a student, I have neither received nor given help on this assignment.}}
\vspace{4mm}

\centerline{Signature: \underline{\parbox[t]{3cm}{\ }}}
\vspace{6mm}

}

\title{\textbf{\Large \mytitle}}
\author{\normalsize{
\ifx\studentid\undefined \student \else \student\ (\studentid) \fi
}}
\date{}

\pagestyle{headings}
\markboth{}{\student}
\renewcommand{\thepage}{Page \arabic{page}, HW \hnumber}

\newenvironment*{sectionprob}[2]{
\noindent {\bf Section #1, Problem #2.}

}{\vspace{8mm}}

\newenvironment*{problem}[1]{
\noindent {\bf Problem #1.}

}{\vspace{8mm}}

\RequirePackage{listings}
\RequirePackage{color}
\lstset{
	tabsize=4,
    basicstyle=\ttfamily\scriptsize,
    %upquote=true,
    aboveskip={1.5\baselineskip},
    columns=fixed,
    showstringspaces=false,
    extendedchars=true,
    breaklines=true,
    prebreak = \raisebox{0ex}[0ex][0ex]{\ensuremath{\hookleftarrow}},
    frame=single,
    rulecolor=\color[rgb]{0.75,0.75,0.75},
    showtabs=false,
    showspaces=false,
    showstringspaces=false,
    keywordstyle=\color[rgb]{0,0,1},
%    numberstyle=\tiny\color[rgb]{0.5,0.5,0.5},
    commentstyle=\color[rgb]{0.133,0.545,0.133},
%    identifierstyle=\color[rgb]{0,0,0},
    stringstyle=\color[rgb]{0.627,0.126,0.941},
}
% usage: \lstinputlisting[language=c++]{<filename>.cpp}
