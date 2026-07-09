import * as React from "react";

export interface DialogProps {
  open: boolean;
  title?: React.ReactNode;
  children: React.ReactNode;
  actions?: React.ReactNode;
  onClose?: () => void;
}

export function Dialog(props: DialogProps): JSX.Element;
