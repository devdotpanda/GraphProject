{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python311.withPackages (ps: with ps; [
      numpy
      matplotlib
      scipy
    ]))
  ];

  shellHook = ''
    echo "Python environment with numpy, matplotlib, and scipy ready."
  '';
}
